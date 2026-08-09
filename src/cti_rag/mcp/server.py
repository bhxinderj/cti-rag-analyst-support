"""
MCP server exposing the CTI-RAG fact pipeline as a single tool.

Fallback cut (Option B1): a FastMCP stdio server with one tool,
``cti_query_tool``, running the *deterministic* portion of the Phase-2
pipeline — query router, hybrid retrieval, entity-aware augmentation,
fact-bundle assembly, severity annotation, and the L1 markdown render.
No LLM is invoked, so a call returns in ~1–2 s and fits comfortably
inside MCP client timeouts; the generative L2 narrative remains the
job of the interactive UI.

Everything is reused import-only from the evaluated pipeline modules;
nothing in the frozen configuration is altered.

Run (Setup B index required):

    CTI_RAG_SETUP=b .venv/bin/python -m src.cti_rag.mcp.server
"""

from __future__ import annotations

import logging
import time

from mcp.server.fastmcp import FastMCP

from ..rag.chain import (
    _assess_context_support,
    _augment_chunks_for_cves,
    _auto_detect_cves_from_chunks,
    _build_chunk_dict,
)
from ..rag.facts import build_fact_bundle
from ..rag.router import classify_query
from ..rag.severity import annotate_vuln_triage_bundle
from ..rag.templates import render_l1
from ..retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

mcp = FastMCP("cti-rag")

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Lazily initialize the hybrid retriever (embedder + reranker load)."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(retrieval_mode="hybrid")
    return _retriever


def run_facts_pipeline(question: str, retriever: HybridRetriever) -> str:
    """Deterministic facts pipeline: route, retrieve, bundle, render L1.

    Mirrors steps 1–6 of ``RAGChain.query_templated`` without the LLM
    generation step. Returns analyst-facing markdown.
    """
    started = time.time()

    decision = classify_query(question)
    chunks = retriever.retrieve(question)
    chunk_dicts = [_build_chunk_dict(chunk) for chunk in chunks]

    primary_entities = list(decision.primary_entities)
    if decision.template == "CrossSourceCompare" and not primary_entities:
        primary_entities = _auto_detect_cves_from_chunks(chunk_dicts)

    if decision.template in ("VulnTriage", "CrossSourceCompare") and primary_entities:
        cve_entities = [e for e in primary_entities if e.upper().startswith("CVE-")]
        if cve_entities:
            chunk_dicts, _ = _augment_chunks_for_cves(retriever, chunk_dicts, cve_entities)

    abstention_reason = _assess_context_support(question, chunk_dicts)
    if abstention_reason:
        elapsed_ms = (time.time() - started) * 1000
        return (
            "## No sufficiently grounded evidence\n\n"
            f"The indexed CTI snapshot cannot answer this question: {abstention_reason}\n\n"
            f"_Deterministic check, no model invoked · {elapsed_ms:.0f} ms_"
        )

    bundle = build_fact_bundle(
        decision.template,
        primary_entities,
        chunk_dicts,
        threat_context_kind=decision.threat_context_kind,
    )
    if decision.template == "VulnTriage":
        for facts in bundle.vuln_triage:
            annotate_vuln_triage_bundle(facts)
    elif decision.template == "CrossSourceCompare" and bundle.cross_source:
        for facts in bundle.cross_source.entities:
            annotate_vuln_triage_bundle(facts)

    l1_block = render_l1(bundle)
    # Presentation-only cleanup for this tool's markdown surface: some KEV
    # metadata carries literal HTML tags which MCP clients render verbatim.
    l1_block = l1_block.replace("<code>", "`").replace("</code>", "`")

    source_lines = "\n".join(
        f"- `{chunk['doc_id']}` ({chunk.get('source', 'unknown')})"
        for chunk in chunk_dicts
    )
    elapsed_ms = (time.time() - started) * 1000

    parts = [part for part in (l1_block,) if part]
    parts.append(f"**Sources ({len(chunk_dicts)})**\n{source_lines}")
    parts.append(
        f"_Template: {decision.template} · deterministic facts, no model invoked · "
        f"{elapsed_ms:.0f} ms_"
    )
    return "\n\n".join(parts)


@mcp.tool()
def cti_query_tool(question: str) -> str:
    """Look up grounded CTI triage facts for a natural-language question.

    Routes the question to a task template (VulnTriage, ThreatContext,
    CrossSourceCompare), retrieves from a curated snapshot of NVD, CISA
    KEV, CISA Advisories, and MISP, and returns a deterministic triage
    card as markdown: CVSS, CWE, affected products, KEV listing,
    ransomware use, a severity signal, and the source document ids.
    Answers come exclusively from the indexed snapshot; if the corpus
    holds no evidence, the tool says so instead of guessing.
    """
    return run_facts_pipeline(question, _get_retriever())


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    mcp.run()
