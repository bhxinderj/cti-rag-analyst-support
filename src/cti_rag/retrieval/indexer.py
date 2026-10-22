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
import shutil
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


def _merge_bm25_artifact(existing: dict, documents: list[CTIDocument]) -> dict:
    """
    Return a new BM25 payload with ``documents`` appended to ``existing``.

    A doc_id that is already present is replaced in place (idempotent
    re-runs of the supplement step), everything else keeps its position so
    the base corpus ordering stays identical to the cloned setup.
    """
    doc_ids = list(existing["doc_ids"])
    corpus_texts = list(existing["corpus_texts"])
    tokenized_corpus = list(existing.get("tokenized_corpus") or [_tokenize_cti(t) for t in corpus_texts])
    metadatas = list(existing.get("metadatas") or [{} for _ in doc_ids])
    position = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    for doc in documents:
        text = doc.to_embedding_text()
        tokens = _tokenize_cti(text)
        meta = doc.to_chromadb_metadata()
        idx = position.get(doc.doc_id)
        if idx is None:
            position[doc.doc_id] = len(doc_ids)
            doc_ids.append(doc.doc_id)
            corpus_texts.append(text)
            tokenized_corpus.append(tokens)
            metadatas.append(meta)
        else:
            corpus_texts[idx] = text
            tokenized_corpus[idx] = tokens
            metadatas[idx] = meta

    return {
        "doc_ids": doc_ids,
        "corpus_texts": corpus_texts,
        "tokenized_corpus": tokenized_corpus,
        "metadatas": metadatas,
    }


def clone_setup_indexes(base_config: dict, target_config: dict, root: Path) -> bool:
    """
    Copy the base setup's persisted indexes into the target setup's paths.

    Returns True when a copy was made, False when the target already exists.
    The ChromaDB collection is renamed to the target's collection name so the
    retriever opens it under the target configuration. The base artifacts
    are only read.
    """
    base_chroma = root / base_config["chromadb"]["persist_directory"]
    base_bm25 = root / base_config["retrieval"]["bm25"]["index_path"]
    target_chroma = root / target_config["chromadb"]["persist_directory"]
    target_bm25 = root / target_config["retrieval"]["bm25"]["index_path"]

    if target_chroma.exists() and target_bm25.exists():
        logger.info("Target indexes already exist at %s; skipping clone", target_bm25.parent)
        return False
    if not base_chroma.exists() or not base_bm25.exists():
        raise FileNotFoundError(
            f"Base setup indexes missing: {base_chroma} / {base_bm25}. Build the base setup first."
        )

    logger.info("Cloning %s -> %s", base_bm25.parent, target_bm25.parent)
    target_bm25.parent.mkdir(parents=True, exist_ok=True)
    if target_chroma.exists():
        shutil.rmtree(target_chroma)
    shutil.copytree(base_chroma, target_chroma)
    shutil.copy2(base_bm25, target_bm25)

    base_name = base_config["retrieval"]["vector"]["collection_name"]
    target_name = target_config["retrieval"]["vector"]["collection_name"]
    if base_name != target_name:
        client = chromadb.PersistentClient(
            path=str(target_chroma),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=base_name)
        collection.modify(name=target_name)
        logger.info("Renamed ChromaDB collection %s -> %s", base_name, target_name)
    return True


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

    def append_documents(self, documents: list[CTIDocument], batch_size: int = 100):
        """
        Add documents to existing indexes without rebuilding the base corpus.

        ChromaDB uses upsert so re-running with the same doc_ids is
        idempotent; the BM25 artifact is merged via _merge_bm25_artifact.
        """
        if not documents:
            logger.warning("No documents to append")
            return
        if not self.bm25_index_path.exists():
            raise FileNotFoundError(
                f"BM25 artifact missing at {self.bm25_index_path}; append needs an existing index."
            )

        logger.info(f"Appending {len(documents)} documents...")
        for i in tqdm(range(0, len(documents), batch_size), desc="ChromaDB (append)"):
            batch = documents[i : i + batch_size]
            self.collection.upsert(
                ids=[doc.doc_id for doc in batch],
                documents=[doc.to_embedding_text() for doc in batch],
                metadatas=[doc.to_chromadb_metadata() for doc in batch],
            )
        logger.info(f"ChromaDB: {self.collection.count()} documents after append")

        merged = _merge_bm25_artifact(_read_bm25_artifact(self.bm25_index_path), documents)
        _write_bm25_artifact(self.bm25_index_path, **merged)
        logger.info(f"BM25 index updated: {len(merged['doc_ids'])} documents")

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
