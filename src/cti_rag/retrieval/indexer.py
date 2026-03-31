"""
Document Indexer.

Takes normalized CTIDocuments and builds both search indexes:
1. ChromaDB vector index (for semantic search)
2. BM25 index (for lexical search)

Both indexes are persisted to disk for reproducibility.
"""

import json
import logging
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import torch
from tqdm import tqdm

from ..ingestion.models import CTIDocument
from ..utils.config import (
    get_project_root,
    load_config,
    require_local_hf_snapshot,
    suppress_noisy_third_party_logs,
)

logger = logging.getLogger(__name__)
suppress_noisy_third_party_logs()


def _resolve_embedding_device(requested_device: str) -> str:
    """Fall back to CPU when the configured accelerator is unavailable."""
    if requested_device != "mps":
        return requested_device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return requested_device

    logger.warning("Configured embedding device 'mps' is unavailable on this host. Falling back to 'cpu'.")
    return "cpu"


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


def _write_bm25_artifact(
    path: Path,
    *,
    doc_ids: list[str],
    corpus_texts: list[str],
    tokenized_corpus: list[list[str]],
    metadatas: list[dict],
) -> None:
    """Persist BM25 inputs as transparent JSON for reproducible rebuilds."""
    payload = {
        "doc_ids": doc_ids,
        "corpus_texts": corpus_texts,
        "tokenized_corpus": tokenized_corpus,
        "metadatas": metadatas,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _read_bm25_artifact(path: Path) -> dict:
    """Load a persisted BM25 JSON artifact."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CTIIndexer:
    """Builds and persists both vector and BM25 indexes."""

    def __init__(self):
        config = load_config()
        self.root = get_project_root()

        # Embedding config
        emb_config = config["embedding"]
        embedding_device = _resolve_embedding_device(emb_config["device"])
        embedding_model_path = require_local_hf_snapshot(
            emb_config["model_name"],
            artifact_label="Embedding model",
        )
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=str(embedding_model_path),
            device=embedding_device,
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

        _write_bm25_artifact(
            self.bm25_index_path,
            doc_ids=doc_ids,
            corpus_texts=corpus_texts,
            tokenized_corpus=tokenized_corpus,
            metadatas=[doc.to_chromadb_metadata() for doc in documents],
        )

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
            bm25_data = _read_bm25_artifact(self.bm25_index_path)
            stats["bm25_count"] = len(bm25_data["doc_ids"])
        return stats
