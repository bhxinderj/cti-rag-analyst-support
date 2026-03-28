"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.

Architecture:
    Query
      |
      +---> BM25 (rank_bm25)          ---> top-k candidates --+
      |                                                         +--> RRF Fusion --> top-n --> Reranker --> top-m
      +---> Vector (ChromaDB)          ---> top-k candidates --+

This module implements the core retrieval pipeline described in the thesis:
- BM25 catches exact CVE-ID matches and technical terms (lexical precision)
- Vector search catches semantic similarity (conceptual relevance)
- RRF combines both without weight tuning (Cormack et al., 2009)
- Cross-encoder reranking adds final precision boost

For ablation study: set retrieval_mode to "bm25", "vector", or "hybrid".
"""

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)
    rank: int = 0


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and vector search with RRF and reranking.

    Supports three modes for ablation study:
    - "hybrid": BM25 + Vector + RRF + Reranking (default, full pipeline)
    - "bm25": BM25 only (lexical baseline)
    - "vector": Vector only (semantic baseline)
    """

    def __init__(self, retrieval_mode: str = "hybrid"):
        config = load_config()
        self.root = get_project_root()
        self.mode = retrieval_mode

        retrieval_config = config["retrieval"]

        # --- Load ChromaDB ---
        chroma_config = config["chromadb"]
        persist_dir = self.root / chroma_config["persist_directory"]

        emb_config = config["embedding"]
        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=emb_config["model_name"],
            device=emb_config["device"],
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_collection(
            name=retrieval_config["vector"]["collection_name"],
            embedding_function=self.embedding_fn,
        )

        # --- Load BM25 ---
        bm25_path = self.root / retrieval_config["bm25"]["index_path"]
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25: BM25Okapi = bm25_data["bm25"]
        self.bm25_doc_ids: list[str] = bm25_data["doc_ids"]
        self.bm25_corpus: list[str] = bm25_data["corpus_texts"]

        # --- Config values ---
        self.bm25_top_k = retrieval_config["bm25"]["top_k"]
        self.vector_top_k = retrieval_config["vector"]["top_k"]
        self.fusion_top_k = retrieval_config["fusion"]["top_k"]
        self.rrf_k = retrieval_config["fusion"]["rrf_k"]
        self.rerank_top_k = retrieval_config["reranker"]["top_k"]

        # --- Load Reranker ---
        reranker_model = retrieval_config["reranker"]["model_name"]
        logger.info(f"Loading reranker: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)

        logger.info(f"HybridRetriever initialized (mode={self.mode})")

    @staticmethod
    def _tokenize_cti(text: str) -> list[str]:
        """CTI-aware tokenizer matching the indexer's tokenization."""
        text = text.lower()
        tokens = text.split()
        cleaned = []
        for token in tokens:
            token = re.sub(r'^[^\w-]+|[^\w-]+$', '', token)
            if token:
                cleaned.append(token)
        return cleaned

    def _search_bm25(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Lexical search using BM25."""
        tokenized_query = self._tokenize_cti(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only include non-zero scores
                results.append(RetrievedChunk(
                    doc_id=self.bm25_doc_ids[idx],
                    content=self.bm25_corpus[idx],
                    score=float(scores[idx]),
                    rank=rank,
                ))
        return results

    def _search_vector(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Semantic search using ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append(RetrievedChunk(
                doc_id=results["ids"][0][i],
                content=results["documents"][0][i],
                score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                rank=i,
            ))
        return chunks

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Combine two ranked lists using Reciprocal Rank Fusion.

        RRF score for document d = sum over all lists L: 1 / (k + rank_L(d))
        where k is a constant (default 60, per Cormack et al., 2009).
        """
        # Build content lookup from both result sets
        content_map: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = {}

        for rank, chunk in enumerate(bm25_results):
            content_map[chunk.doc_id] = chunk
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, chunk in enumerate(vector_results):
            if chunk.doc_id not in content_map:
                content_map[chunk.doc_id] = chunk
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        fused = []
        for rank, doc_id in enumerate(sorted_ids[: self.fusion_top_k]):
            chunk = content_map[doc_id]
            chunk.score = rrf_scores[doc_id]
            chunk.rank = rank
            fused.append(chunk)

        return fused

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank chunks using cross-encoder model."""
        if not chunks:
            return chunks

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)

        for rank, chunk in enumerate(reranked):
            chunk.rank = rank

        return reranked[: self.rerank_top_k]

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Execute the full retrieval pipeline.

        Returns top-k chunks based on the configured retrieval mode.
        """
        logger.debug(f"Retrieving for query: {query[:80]}... (mode={self.mode})")

        if self.mode == "bm25":
            results = self._search_bm25(query, self.rerank_top_k)

        elif self.mode == "vector":
            results = self._search_vector(query, self.rerank_top_k)

        elif self.mode == "hybrid":
            bm25_results = self._search_bm25(query, self.bm25_top_k)
            vector_results = self._search_vector(query, self.vector_top_k)

            logger.debug(f"BM25: {len(bm25_results)} results, Vector: {len(vector_results)} results")

            fused = self._reciprocal_rank_fusion(bm25_results, vector_results)
            results = self._rerank(query, fused)

        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

        logger.info(f"Retrieved {len(results)} chunks (mode={self.mode})")
        return results
