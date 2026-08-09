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
from ..rag.conversation import ConversationState, resolve_question
from ..rag.extended import ExtendedChain
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
_extended_chain: ExtendedChain | None = None


def _get_chain(mode: str) -> RAGChain:
    if mode not in _chains:
        _chains[mode] = RAGChain(retrieval_mode=mode)
    return _chains[mode]


def _get_extended_chain() -> ExtendedChain:
    global _extended_chain
    if _extended_chain is None:
        try:
            _extended_chain = ExtendedChain()
        except RuntimeError as exc:
            # Typically: missing OPENROUTER_API_KEY in the server env.
            raise HTTPException(status_code=503, detail=str(exc))
    return _extended_chain


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    generation_model: str | None = None
    try:
        if req.extended:
            extended = _get_extended_chain()
            generation_model = extended.generation_model_label
            condense_llm = extended.llm
        else:
            chain = _get_chain(req.mode)
            # Local mode condenses locally — nothing leaves the machine.
            condense_llm = chain.llm

        state = ConversationState(
            last_question=req.last_question, last_entities=list(req.last_entities)
        )
        resolved = await asyncio.to_thread(
            resolve_question, req.question, state, condense_llm
        )
        pipeline_question = resolved.question

        if req.extended:
            response = await asyncio.to_thread(extended.query, pipeline_question)
        elif req.templated:
            response = await asyncio.to_thread(chain.query_templated, pipeline_question)
        else:
            response = await asyncio.to_thread(chain.query, pipeline_question)
    except HTTPException:
        raise
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
        query=req.question,
        answer=response.answer,
        sources=sources,
        retrieval_mode=response.retrieval_mode,
        retrieval_time_ms=round(response.retrieval_time_ms, 1),
        generation_time_ms=round(response.generation_time_ms, 1),
        total_time_ms=round(response.total_time_ms, 1),
        grounded=response.abstention_reason is None,
        abstention_reason=response.abstention_reason,
        grounding_warnings=response.grounding_warnings,
        pipeline="extended" if req.extended else ("templated" if req.templated else "legacy"),
        generation_model=generation_model,
        resolved_question=(
            resolved.question if resolved.method != "passthrough" else None
        ),
        resolution_method=(
            resolved.method if resolved.method != "passthrough" else None
        ),
        template=response.template or None,
        routing_decision=response.routing_decision or None,
        l1_block=response.l1_block or None,
        l2_output=response.l2_output or None,
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
