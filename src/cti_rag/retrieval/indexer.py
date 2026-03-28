"""
Document Indexer.

Takes normalized CTIDocuments and builds both search indexes:
1. ChromaDB vector index (for semantic search)
2. BM25 index (for lexical search)

Both indexes are persisted to disk for reproducibility.
"""

import logging
import pickle
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ..ingestion.models import CTIDocument
from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)


def _tokenize_cti(text: str) -> list[str]:
    """
    CTI-aware tokenizer for BM25.

    Handles CTI-specific identifiers correctly:
    - Strips trailing punctuation (so queries like "CVE-2021-44228?" work)
    - Preserves hyphenated CTI identifiers as single tokens (CVE-2021-44228, CWE-79)
    - Lowercases everything for case-insensitive matching
    """
    text = text.lower()
    # Split on whitespace, then strip trailing punctuation from each token
    tokens = text.split()
    cleaned = []
    for token in tokens:
        # Strip punctuation from edges but preserve hyphens inside tokens
        token = re.sub(r'^[^\w-]+|[^\w-]+$', '', token)
        if token:
            cleaned.append(token)
    return cleaned


class CTIIndexer:
    """Builds and persists both vector and BM25 indexes."""

    def __init__(self):
        config = load_config()
        self.root = get_project_root()

        # Embedding config
        emb_config = config["embedding"]
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=emb_config["model_name"],
            device=emb_config["device"],
        )

        # ChromaDB config
        chroma_config = config["chromadb"]
        persist_dir = self.root / chroma_config["persist_directory"]
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

        collection_name = config["retrieval"]["vector"]["collection_name"]
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": config["retrieval"]["vector"]["similarity_metric"]},
        )

        # BM25 config
        self.bm25_index_path = self.root / config["retrieval"]["bm25"]["index_path"]
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

    def index_documents(self, documents: list[CTIDocument], batch_size: int = 100):
        """
        Index all documents into both ChromaDB and BM25.

        Args:
            documents: List of normalized CTIDocuments
            batch_size: ChromaDB insertion batch size
        """
        if not documents:
            logger.warning("No documents to index")
            return

        logger.info(f"Indexing {len(documents)} documents...")

        # --- Build ChromaDB vector index ---
        logger.info("Building ChromaDB vector index...")
        for i in tqdm(range(0, len(documents), batch_size), desc="ChromaDB"):
            batch = documents[i : i + batch_size]
            self.collection.add(
                ids=[doc.doc_id for doc in batch],
                documents=[doc.to_embedding_text() for doc in batch],
                metadatas=[doc.to_chromadb_metadata() for doc in batch],
            )

        logger.info(f"ChromaDB: {self.collection.count()} documents indexed")

        # --- Build BM25 index ---
        logger.info("Building BM25 index...")
        corpus_texts = [doc.to_embedding_text() for doc in documents]
        doc_ids = [doc.doc_id for doc in documents]

        # Tokenize for BM25 with CTI-aware preprocessing:
        # - Strip punctuation so "CVE-2021-44228?" matches "CVE-2021-44228"
        # - Preserve hyphenated identifiers (CVE-IDs, CWE-IDs, ATT&CK IDs)
        tokenized_corpus = [_tokenize_cti(text) for text in corpus_texts]

        bm25 = BM25Okapi(tokenized_corpus)

        # Persist BM25 index + mapping
        bm25_data = {
            "bm25": bm25,
            "doc_ids": doc_ids,
            "corpus_texts": corpus_texts,
        }
        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(bm25_data, f)

        logger.info(f"BM25 index saved: {self.bm25_index_path}")
        logger.info(f"Indexing complete: {len(documents)} documents in both indexes")

    def clear_indexes(self):
        """Remove all documents from both indexes. Use with caution."""
        collection_name = self.collection.name
        self.chroma_client.delete_collection(collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        if self.bm25_index_path.exists():
            self.bm25_index_path.unlink()

        logger.info("All indexes cleared")

    def get_stats(self) -> dict:
        """Return index statistics."""
        stats = {
            "chromadb_count": self.collection.count(),
            "bm25_exists": self.bm25_index_path.exists(),
        }
        if self.bm25_index_path.exists():
            with open(self.bm25_index_path, "rb") as f:
                bm25_data = pickle.load(f)
            stats["bm25_count"] = len(bm25_data["doc_ids"])
        return stats
