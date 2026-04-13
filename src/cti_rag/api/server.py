"""
CTI-RAG API Server.

Thin HTTP layer over the RAG pipeline for the demo UI.

    uvicorn src.cti_rag.api.server:app --port 8000
"""

import asyncio
import logging

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..rag.chain import RAGChain
from ..utils.config import load_config, get_project_root
from .schemas import HealthResponse, QueryRequest, QueryResponse, SourceDocument

logger = logging.getLogger(__name__)

app = FastAPI(title="CTI-RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_chains: dict[str, RAGChain] = {}


def _get_chain(mode: str) -> RAGChain:
    if mode not in _chains:
        _chains[mode] = RAGChain(retrieval_mode=mode)
    return _chains[mode]


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    chain = _get_chain(req.mode)
    try:
        response = await asyncio.to_thread(chain.query, req.question)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceDocument(
            doc_id=doc.get("doc_id", ""),
            title=doc.get("title", ""),
            source=doc.get("source", "unknown"),
            score=round(doc.get("score", 0.0), 4),
            rank=doc.get("rank", 0),
            citation_label=doc.get("citation_label", ""),
            cited_in_answer=doc.get("cited_in_answer", False),
        )
        for doc in response.source_documents
    ]

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        sources=sources,
        retrieval_mode=response.retrieval_mode,
        retrieval_time_ms=round(response.retrieval_time_ms, 1),
        generation_time_ms=round(response.generation_time_ms, 1),
        total_time_ms=round(response.total_time_ms, 1),
        grounded=response.abstention_reason is None,
        abstention_reason=response.abstention_reason,
        grounding_warnings=response.grounding_warnings,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health():
    config = load_config()
    root = get_project_root()
    active_setup = config["data"].get("active_setup", "default")

    ollama_reachable = False
    try:
        resp = requests.get(config["llm"]["base_url"], timeout=3)
        ollama_reachable = resp.status_code == 200
    except Exception:
        pass

    chroma_dir = root / config["chromadb"]["persist_directory"]
    bm25_path = root / config["retrieval"]["bm25"]["index_path"]
    index_loaded = chroma_dir.exists() and bm25_path.exists()

    status = "ok" if (ollama_reachable and index_loaded) else "degraded"

    return HealthResponse(
        status=status,
        ollama_reachable=ollama_reachable,
        index_loaded=index_loaded,
        active_setup=active_setup,
    )
