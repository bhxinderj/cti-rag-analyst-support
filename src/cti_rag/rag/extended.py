"""
Extended-Analysis mode: hosted generation over the unchanged Phase-2
fact pipeline (Option B3).

This mode is a deliberate, clearly-flagged user choice and NOT part of
the evaluated configuration (tag ``evaluated-v1``): question and
retrieved context are sent to a hosted model, in exchange for
substantially richer analyst assistance and ~10x lower latency (the
local-vs-hosted trade-off is quantified by the hosted ablation row in
the thesis results).

Differences to the evaluated templated pipeline — all additive,
everything else is reused import-only:

  1. Generation via the hosted provider from ``build_generation_llm``
     (OpenRouter; deliberately not the evaluation judge model).
  2. Deeper retrieval (top-8 instead of top-5 reranked chunks): the
     advisory corpus carries hunting-guidance and incident-response
     sections that rarely fit a top-5 context.
  3. An analyst-assistance instruction layer appended to the unchanged
     template prompt: prioritized mitigations, detection/hunting
     starting points, and status-quo checks — still strictly grounded
     in the retrieved context with citations.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.hybrid_retriever import HybridRetriever
from .chain import (
    RAGResponse,
    _assess_context_support,
    _augment_chunks_for_cves,
    _auto_detect_cves_from_chunks,
    _build_abstention_answer,
    _build_chunk_dict,
    _extract_citation_labels,
    _normalize_response_citations,
    build_generation_llm,
)
from .facts import build_fact_bundle
from .router import classify_query
from .severity import annotate_vuln_triage_bundle
from .templates import build_prompt as build_template_prompt, render_l1
from ..utils.config import load_config

logger = logging.getLogger(__name__)

_EXTENDED_RERANK_TOP_K = 8

_ANALYST_ASSIST_INSTRUCTIONS = (
    "\n\nExtended analyst assistance (this mode only):\n"
    "- In **Mitigations**, order actions by urgency and be operationally "
    "concrete: name the action, the affected component, and the deadline "
    "or version where the context provides one.\n"
    "- Add a **Detection & Hunting** section after Mitigations when the "
    "retrieved context contains detection guidance, hunting queries, log "
    "sources, or IoCs: give the analyst concrete starting points (what to "
    "search, where to look, which artifacts confirm compromise).\n"
    "- Add a **Next Checks** section listing 2-4 short, concrete steps an "
    "analyst should take to establish the status quo in their own "
    "environment (e.g. version checks, session audits, KEV due-date "
    "comparison).\n"
    "- Every claim in these sections must still be grounded in the "
    "retrieved context and carry a citation label. Do not invent "
    "environment-specific details; phrase Next Checks as instructions, "
    "not as facts about the analyst's environment."
)


class ExtendedChain:
    """Hosted-generation analyst-assist chain over the frozen fact pipeline."""

    def __init__(self):
        config = load_config()
        llm_config = dict(config["llm"])
        llm_config["provider"] = "openrouter"
        self.generation_model_label, self.llm = build_generation_llm(llm_config)

        self.retriever = HybridRetriever(retrieval_mode="hybrid")
        # Instance-level override only — the frozen module and the shared
        # config keep their evaluated top-5 default.
        self.retriever.rerank_top_k = _EXTENDED_RERANK_TOP_K

        logger.info(
            "ExtendedChain initialized (generation=%s, rerank_top_k=%d)",
            self.generation_model_label,
            _EXTENDED_RERANK_TOP_K,
        )

    def query(self, question: str) -> RAGResponse:
        """Mirror of the templated pipeline with hosted analyst-assist L2."""
        total_start = time.time()

        decision = classify_query(question)

        retrieval_start = time.time()
        chunks = self.retriever.retrieve(question)
        retrieval_time = (time.time() - retrieval_start) * 1000
        chunk_dicts = [_build_chunk_dict(chunk) for chunk in chunks]

        primary_entities = list(decision.primary_entities)
        if decision.template == "CrossSourceCompare" and not primary_entities:
            primary_entities = _auto_detect_cves_from_chunks(chunk_dicts)

        if decision.template in ("VulnTriage", "CrossSourceCompare") and primary_entities:
            cve_entities = [e for e in primary_entities if e.upper().startswith("CVE-")]
            if cve_entities:
                chunk_dicts, _ = _augment_chunks_for_cves(
                    self.retriever, chunk_dicts, cve_entities
                )

        abstention_reason = _assess_context_support(question, chunk_dicts)
        if abstention_reason:
            total_time = (time.time() - total_start) * 1000
            return RAGResponse(
                query=question,
                answer=_build_abstention_answer(abstention_reason),
                contexts=[c.get("content", "") for c in chunk_dicts],
                source_documents=chunk_dicts,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0,
                total_time_ms=total_time,
                retrieval_mode="hybrid",
                abstention_reason=abstention_reason,
                template=decision.template,
                routing_decision=decision.as_trace_dict(),
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

        messages_dict = build_template_prompt(bundle, question, chunk_dicts)
        # Analyst-assist layer: appended to the unchanged template system
        # prompt — templates.py stays frozen.
        messages = []
        for m in messages_dict:
            content = m["content"]
            if m["role"] == "system":
                content = content + _ANALYST_ASSIST_INSTRUCTIONS
                messages.append(SystemMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        generation_start = time.time()
        response = self.llm.invoke(messages)
        generation_time = (time.time() - generation_start) * 1000

        raw_llm_output = response.content
        l2_output = _normalize_response_citations(raw_llm_output, chunk_dicts)

        final_answer = "\n\n".join(part for part in (l1_block, l2_output) if part)

        cited_labels = _extract_citation_labels(l2_output)
        for chunk_dict in chunk_dicts:
            chunk_dict["cited_in_answer"] = chunk_dict["citation_label"] in cited_labels

        total_time = (time.time() - total_start) * 1000
        logger.info(
            "Extended generation in %.0fms (total: %.0fms)", generation_time, total_time
        )

        return RAGResponse(
            query=question,
            answer=final_answer,
            contexts=[c.get("content", "") for c in chunk_dicts],
            source_documents=chunk_dicts,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieval_mode="hybrid",
            raw_llm_output=raw_llm_output,
            template=decision.template,
            routing_decision=decision.as_trace_dict(),
            l1_block=l1_block,
            l2_output=l2_output,
        )
