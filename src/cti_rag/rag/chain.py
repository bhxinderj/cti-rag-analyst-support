"""
RAG Orchestration Chain.

Wires together: Query -> Hybrid Retrieval -> Rerank -> Context Assembly -> LLM -> Response.
Uses LangChain LCEL for composability and transparency.

This is the main entry point for the RAG pipeline. Each step is explicitly
logged and inspectable for evaluation and debugging.
"""

import logging
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from ..utils.config import load_config
from .facts import build_fact_bundle, extract_cve_ids
from .prompts import (
    SYSTEM_PROMPT,
    format_context,
    QUERY_TEMPLATE,
    build_citation_label,
    build_baseline_prompt,
)
from .router import RoutingDecision, classify_query
from .severity import annotate_vuln_triage_bundle
from .templates import build_prompt as build_template_prompt, render_l1

logger = logging.getLogger(__name__)

_CTI_ENTITY_PATTERNS = (
    re.compile(r"\bCVE-\d{4}-\d{4,}\b", flags=re.IGNORECASE),
    re.compile(r"\bCWE-\d+\b", flags=re.IGNORECASE),
    re.compile(r"\bT\d{4}(?:\.\d{3})?\b", flags=re.IGNORECASE),
    re.compile(r"\bTA\d{4}\b", flags=re.IGNORECASE),
    re.compile(r"\b[GSM]\d{4}\b", flags=re.IGNORECASE),
    re.compile(r"\bDS\d{4}\b", flags=re.IGNORECASE),
    re.compile(r"\bDET\d{4}\b", flags=re.IGNORECASE),
)
_GROUNDING_STOPWORDS = {
    "about", "after", "against", "also", "among", "been", "being", "does",
    "from", "have", "into", "known", "more", "most", "that", "their",
    "them", "then", "they", "this", "those", "used", "using", "what",
    "when", "where", "which", "with", "within", "would", "could", "should",
    "there", "these", "than", "into", "across", "query", "question",
}
_CITATION_BLOCK_RE = re.compile(r"\[(?:Source|Sources)\s*:\s*([^\]]+)\]")
# Short-form citation: ``[doc_id]`` without the ``Source:`` prefix.  Accepted
# only when the bracketed text resolves to a known citation label or doc_id,
# so plain entity references like ``[CVE-2024-3094]`` are not misread as
# sources.
_SHORT_CITATION_RE = re.compile(r"\[([^\[\]\s][^\[\]]*?)\]")
_ABSTENTION_SUMMARY = "Insufficient evidence in the retrieved context to answer this question."
_SUMMARY_PREFIX = "Summary:"
_WHY_SECTION_HEADER = "Why it matters:"
_MITIGATION_SECTION_HEADER = "Recommended actions / Mitigations:"
_EVIDENCE_SECTION_HEADER = "Evidence:"
_SUMMARY_FALLBACK = "Source-grounded summary could not be preserved after citation checks."
_WHY_FALLBACK = "No clearly source-grounded impact statement could be preserved from the generated answer."
_MITIGATION_FALLBACK = "No reliable mitigation guidance is present in the retrieved context."
_EVIDENCE_FALLBACK = "No clearly source-grounded evidence statements could be preserved from the generated answer."
_STRUCTURED_SECTION_HEADERS = {
    _SUMMARY_PREFIX,
    _WHY_SECTION_HEADER,
    _MITIGATION_SECTION_HEADER,
    _EVIDENCE_SECTION_HEADER,
    "Unknowns / Gaps:",
    "Missing:",
}


def _normalize_alias(text: str) -> str:
    """Normalize citation aliases and free-text matches for exact lookup."""
    return " ".join(text.split()).strip().lower()


def _extract_query_entities(query: str) -> list[str]:
    """Extract exact CTI identifiers that should be present in relevant context."""
    entities = []
    for pattern in _CTI_ENTITY_PATTERNS:
        entities.extend(match.group(0).lower() for match in pattern.finditer(query))
    return sorted(set(entities))


def _extract_meaningful_terms(text: str) -> list[str]:
    """Keep query terms that can act as a cheap relevance signal."""
    tokens = re.findall(r"[a-z0-9_-]+", text.lower())
    return sorted({
        token for token in tokens
        if len(token) >= 4 and token not in _GROUNDING_STOPWORDS and not token.isdigit()
    })


def _build_abstention_answer(reason: str) -> str:
    """Return the analyst-facing abstention text."""
    return f"Summary: {_ABSTENTION_SUMMARY}\nMissing: {reason}"


def _assess_context_support(question: str, chunk_dicts: list[dict]) -> str | None:
    """
    Decide whether the retrieved context is strong enough to justify generation.

    The guard is intentionally simple:
    - no context => abstain
    - CTI identifiers in query must appear in retrieved evidence
    - otherwise require at least minimal lexical overlap with the question
    """
    if not chunk_dicts:
        return "No relevant context was retrieved."

    query_entities = _extract_query_entities(question)
    if query_entities:
        combined_context = " ".join(
            f"{chunk.get('doc_id', '')} {chunk.get('title', '')} {chunk.get('content', '')}".lower()
            for chunk in chunk_dicts
        )
        missing_entities = [entity for entity in query_entities if entity not in combined_context]
        if missing_entities:
            return f"The retrieved context does not mention {', '.join(missing_entities)}."
        return None

    query_terms = _extract_meaningful_terms(question)
    if not query_terms:
        return None

    total_matches: set[str] = set()
    best_chunk_matches = 0

    for chunk in chunk_dicts:
        haystack = f"{chunk.get('title', '')} {chunk.get('content', '')}".lower()
        chunk_matches = {term for term in query_terms if term in haystack}
        total_matches.update(chunk_matches)
        best_chunk_matches = max(best_chunk_matches, len(chunk_matches))

    required_matches = min(2, len(query_terms))
    if best_chunk_matches < required_matches and len(total_matches) < required_matches:
        return "The retrieved context is only weakly related to the question."

    return None


_CTI_ID_IN_ALIAS_RE = re.compile(
    r"(CVE-\d{4}-\d{4,}|CWE-\d+|T\d{4}(?:\.\d{3})?|TA\d{4}|[GSM]\d{4}|DS\d{4}|DET\d{4})",
    flags=re.IGNORECASE,
)
# Tokens below this length are too noisy for fuzzy matching ("the", "log4j"
# is fine, "nvd" is not — it matches every nvd_* chunk).
_FUZZY_MIN_TOKEN_LEN = 5


def _alias_variants(text: str) -> list[str]:
    """Expand a raw alias string into additional normalized variants.

    The retrieval layer labels chunks like ``"Log4Shell | nvd_CVE-2021-44228"``
    or ``"nvd_CVE-2024-3094"``; LLMs frequently cite them as plain
    ``CVE-2021-44228``, which then fails the strict alias_map lookup and
    gets stripped. This helper pulls out CTI identifiers and bare doc-id
    suffixes so those forms still resolve deterministically.
    """
    variants: list[str] = []
    stripped = (text or "").strip()
    if not stripped:
        return variants

    variants.append(stripped)

    # Bare doc_id without source-type prefix, e.g. "nvd_CVE-2024-3094" -> "CVE-2024-3094".
    if "_" in stripped:
        suffix = stripped.split("_", 1)[1].strip()
        if suffix:
            variants.append(suffix)

    # Extract embedded CTI identifiers (CVE, CWE, ATT&CK technique/tactic, ...).
    for match in _CTI_ID_IN_ALIAS_RE.finditer(stripped):
        variants.append(match.group(1))

    return variants


def _build_citation_alias_map(
    source_documents: list[dict],
) -> tuple[dict[str, str], dict[str, str], dict[int, str]]:
    """Map allowed citation aliases back to their canonical citation label.

    In addition to exact aliases (canonical, doc_id, title, and their
    :func:`_alias_variants` expansions), each ``doc_id`` is stored in a
    secondary map so that :func:`_resolve_citation` can fall back to
    substring matching when the LLM embeds a ``doc_id`` inside a longer
    free-text citation.  A position map (1-indexed) is also returned so
    that numeric citations like ``[1]`` or ``[2]`` — referring to the
    Nth chunk shown in the prompt context — can be resolved.
    """
    alias_map: dict[str, str] = {}
    ambiguous_aliases: set[str] = set()
    doc_id_map: dict[str, str] = {}
    position_map: dict[int, str] = {}

    for index, doc in enumerate(source_documents, start=1):
        canonical = doc.get("citation_label", "").strip()
        if not canonical:
            continue

        position_map[index] = canonical

        doc_id = doc.get("doc_id", "").strip()
        if doc_id:
            doc_id_map[doc_id.lower()] = canonical

        raw_sources = (canonical, doc_id, doc.get("title", ""))
        expanded: list[str] = []
        for raw in raw_sources:
            expanded.extend(_alias_variants(raw))

        for alias in expanded:
            normalized = _normalize_alias(alias)
            if not normalized:
                continue

            existing = alias_map.get(normalized)
            if existing and existing != canonical:
                ambiguous_aliases.add(normalized)
                continue

            alias_map[normalized] = canonical

    for alias in ambiguous_aliases:
        alias_map.pop(alias, None)

    return alias_map, doc_id_map, position_map


def _resolve_citation(
    candidate: str,
    alias_map: dict[str, str],
    doc_id_map: dict[str, str],
    position_map: dict[int, str] | None = None,
) -> str | None:
    """Resolve a single citation candidate to its canonical label.

    1. Exact alias lookup (fast path, includes :func:`_alias_variants`).
    2. Numeric fallback: if the candidate is a pure integer referring to
       a valid context position, resolve to that chunk's canonical label.
       Llama-8B occasionally cites context chunks by their ``[N]`` index
       (the numbered list used in :func:`format_context`).
    3. Substring fallback: if the candidate contains a known ``doc_id``,
       accept it.  This covers cases where the LLM wraps the doc_id in
       extra text (e.g. ``"NVD entry nvd_CVE-2024-3094"``).
    """
    canonical = alias_map.get(_normalize_alias(candidate))
    if canonical:
        return canonical

    candidate_stripped = candidate.strip()
    if position_map and candidate_stripped.isdigit():
        try:
            idx = int(candidate_stripped)
        except ValueError:
            idx = 0
        if idx in position_map:
            return position_map[idx]

    candidate_lower = candidate.lower()
    for doc_id_lower, canon in doc_id_map.items():
        if doc_id_lower in candidate_lower:
            return canon

    return None


def _fuzzy_resolve_alias(candidate: str, source_documents: list[dict]) -> str | None:
    """Best-effort fallback when the strict alias map misses.

    Strategy, ordered by specificity (first deterministic hit wins):

    1. Substring match of ``candidate`` inside any canonical citation
       label or doc_id — recovers cases like ``"Log4Shell"`` cited when
       the canonical label is ``"Log4Shell CVE | nvd_CVE-2021-44228"``.
    2. Token overlap with the chunk title: the candidate must contribute
       at least two substantive tokens (length >= ``_FUZZY_MIN_TOKEN_LEN``)
       that also appear in the title.

    Returns ``None`` when no chunk is unambiguously identified.
    """
    needle = _normalize_alias(candidate)
    if not needle:
        return None

    # Stage 1: substring match on canonical label or doc_id.
    substring_hits: list[str] = []
    for doc in source_documents:
        canonical = doc.get("citation_label", "").strip()
        if not canonical:
            continue
        haystacks = [
            _normalize_alias(canonical),
            _normalize_alias(doc.get("doc_id", "")),
        ]
        if any(needle and needle in h for h in haystacks):
            if canonical not in substring_hits:
                substring_hits.append(canonical)
    if len(substring_hits) == 1:
        return substring_hits[0]

    # Stage 2: token overlap with title.
    candidate_tokens = {
        tok for tok in re.split(r"[^a-z0-9]+", needle)
        if len(tok) >= _FUZZY_MIN_TOKEN_LEN
    }
    if len(candidate_tokens) < 2:
        return None

    title_hits: list[str] = []
    for doc in source_documents:
        canonical = doc.get("citation_label", "").strip()
        title = _normalize_alias(doc.get("title", ""))
        if not canonical or not title:
            continue
        title_tokens = {
            tok for tok in re.split(r"[^a-z0-9]+", title)
            if len(tok) >= _FUZZY_MIN_TOKEN_LEN
        }
        if len(candidate_tokens & title_tokens) >= 2 and canonical not in title_hits:
            title_hits.append(canonical)
    if len(title_hits) == 1:
        return title_hits[0]

    return None


def _normalize_response_citations(answer: str, source_documents: list[dict]) -> str:
    """
    Normalize citation blocks to the canonical [Source: <citation_label>] form.

    Handles two bracket styles:
      * ``[Source: <label>]`` — the prescribed form from the system prompt.
        Resolution per candidate: exact alias map (incl.
        :func:`_alias_variants`), numeric ``[N]`` position, doc_id
        substring, then :func:`_fuzzy_resolve_alias`. Unresolvable
        individual candidates are dropped; valid ones in the block survive.
      * ``[<label>]`` — short form the model frequently uses with terse
        ``doc_id``-style labels. Accepted only when the bracketed text
        resolves deterministically (no fuzzy stage), so inline entity
        references like ``[CVE-2024-3094]`` embedded in prose do not get
        rewritten and unknown tokens stay untouched.
    """
    alias_map, doc_id_map, position_map = _build_citation_alias_map(source_documents)
    if not alias_map and not doc_id_map and not position_map:
        return answer

    def resolve_block(raw_value: str, allow_fuzzy: bool) -> list[str]:
        candidates = [raw_value]
        if ";" in raw_value:
            candidates = [part.strip() for part in raw_value.split(";") if part.strip()]
        elif "," in raw_value:
            candidates = [part.strip() for part in raw_value.split(",") if part.strip()]

        canonical_labels: list[str] = []
        for candidate in candidates:
            canonical = _resolve_citation(candidate, alias_map, doc_id_map, position_map)
            if not canonical and allow_fuzzy:
                canonical = _fuzzy_resolve_alias(candidate, source_documents)
            if canonical and canonical not in canonical_labels:
                canonical_labels.append(canonical)
        return canonical_labels

    def replace_source(match: re.Match) -> str:
        # Explicit [Source: ...] blocks carry unambiguous citation intent,
        # so the fuzzy fallback may be used. Unresolvable individual
        # candidates are dropped; valid ones in the same block survive.
        canonical_labels = resolve_block(match.group(1).strip(), allow_fuzzy=True)
        if not canonical_labels:
            return ""
        return " ".join(f"[Source: {label}]" for label in canonical_labels)

    def replace_short(match: re.Match) -> str:
        raw_value = match.group(1).strip()
        # Skip blocks that already use the ``Source:`` prefix — those were
        # handled in the first pass.
        if re.match(r"(?i)^sources?\s*:", raw_value):
            return match.group(0)
        # Short-form brackets resolve deterministically only — fuzzy
        # matching would turn prose entity mentions into citations.
        canonical_labels = resolve_block(raw_value, allow_fuzzy=False)
        if not canonical_labels:
            return match.group(0)  # keep original text untouched
        return " ".join(f"[Source: {label}]" for label in canonical_labels)

    normalized = _CITATION_BLOCK_RE.sub(replace_source, answer)
    normalized = _SHORT_CITATION_RE.sub(replace_short, normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _extract_citation_labels(answer: str) -> set[str]:
    """Extract canonical citation labels that survived normalization."""
    labels: set[str] = set()
    for match in _CITATION_BLOCK_RE.finditer(answer):
        raw_value = match.group(1).strip()
        if raw_value:
            labels.add(raw_value)
    return labels


# NOTE (Step 12 first iteration, rolled back 2026-04-15):
# An earlier revision of this module shipped an ``_enforce_l2_citations``
# helper that rewrote uncited L2 claim lines into explicit grounding
# notes, on the assumption that Phase-2's low citation_coverage (~0.09)
# was caused by aggressive alias stripping. The live smoke on the three
# calibrated queries (vuln_004, ttp_001, cross_003) surfaced a different
# root cause: the local Ollama model rarely emits ``[Source: ...]`` at
# all, preferring to dump chunk labels as a flat list under an
# ``**Evidence**`` section. Line-level enforcement therefore replaced
# *every* claim line with a grounding note and crashed the Rubric
# Mean-Total from ~2.5 to 1.17 (traceability 0.33). The feature was
# reverted; the real fix lives in the template prompts (one-shot
# citation example, output-format constraints) and is scoped as a
# follow-up iteration. Fuzzy alias resolution (_alias_variants /
# _fuzzy_resolve_alias / _normalize_response_citations) and the
# hardened system prompt survive from Step 12 — they are strictly
# additive and do no harm when the LLM abstains from inline citations.


def _enforce_summary_grounding(answer: str) -> str:
    """Replace an uncited inline summary with a transparent fallback."""
    lines = answer.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(_SUMMARY_PREFIX):
            continue

        summary_content = stripped[len(_SUMMARY_PREFIX):].strip()
        if not summary_content:
            return answer
        if _CITATION_BLOCK_RE.search(line):
            return answer

        lines[index] = f"{_SUMMARY_PREFIX} {_SUMMARY_FALLBACK}"
        return "\n".join(lines).strip()

    return answer


def _filter_section_to_cited_lines(answer: str, section_header: str, fallback_line: str) -> str:
    """
    Keep only cited lines in a structured section.

    This is intentionally conservative: uncited lines are removed instead of being
    left in place with misleading source confidence.
    """
    if section_header not in answer:
        return answer

    lines = answer.splitlines()
    output_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        output_lines.append(line)

        if line.strip() != section_header:
            i += 1
            continue

        i += 1
        section_lines: list[str] = []
        while i < len(lines) and lines[i].strip() not in _STRUCTURED_SECTION_HEADERS:
            section_lines.append(lines[i])
            i += 1

        grounded_lines = [
            section_line
            for section_line in section_lines
            if _CITATION_BLOCK_RE.search(section_line)
        ]

        while grounded_lines and not grounded_lines[0].strip():
            grounded_lines.pop(0)
        while grounded_lines and not grounded_lines[-1].strip():
            grounded_lines.pop()

        if grounded_lines:
            output_lines.extend(grounded_lines)
        else:
            output_lines.append(fallback_line)

        if i < len(lines):
            continue

    cleaned = "\n".join(output_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _enforce_mitigation_grounding(answer: str) -> str:
    """Keep mitigation guidance only when it remains source-grounded."""
    return _filter_section_to_cited_lines(answer, _MITIGATION_SECTION_HEADER, _MITIGATION_FALLBACK)


def _enforce_why_grounding(answer: str) -> str:
    """Keep only cited impact lines in the Why it matters section."""
    return _filter_section_to_cited_lines(answer, _WHY_SECTION_HEADER, _WHY_FALLBACK)


def _enforce_evidence_grounding(answer: str) -> str:
    """Keep only cited evidence bullets so the Evidence section stays meaningful."""
    return _filter_section_to_cited_lines(answer, _EVIDENCE_SECTION_HEADER, _EVIDENCE_FALLBACK)


def _analyze_citation_compliance(answer: str) -> dict[str, int | bool]:
    """Collect lightweight citation-compliance signals from the final answer."""
    current_section: str | None = None
    cited_lines = 0
    uncited_lines = 0
    evidence_cited_lines = 0
    has_evidence_section = False

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped in _STRUCTURED_SECTION_HEADERS:
            current_section = stripped
            if stripped == _EVIDENCE_SECTION_HEADER:
                has_evidence_section = True
            continue

        if current_section == "Unknowns / Gaps:":
            continue
        if stripped in {_SUMMARY_FALLBACK, _WHY_FALLBACK, _MITIGATION_FALLBACK, _EVIDENCE_FALLBACK}:
            continue

        has_citation = bool(_CITATION_BLOCK_RE.search(line))
        if has_citation:
            cited_lines += 1
            if current_section == _EVIDENCE_SECTION_HEADER:
                evidence_cited_lines += 1
        else:
            uncited_lines += 1

    return {
        "cited_lines": cited_lines,
        "uncited_lines": uncited_lines,
        "evidence_cited_lines": evidence_cited_lines,
        "has_evidence_section": has_evidence_section,
    }


def _append_grounding_notes(answer: str, notes: list[str]) -> str:
    """Append transparency notes to Unknowns / Gaps without duplicating them."""
    unique_notes: list[str] = []
    for note in notes:
        if note and note not in unique_notes and note not in answer:
            unique_notes.append(note)

    if not unique_notes:
        return answer

    lines = answer.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.strip() == "Unknowns / Gaps:"),
        None,
    )
    note_lines = [f"- Grounding note: {note}" for note in unique_notes]

    if header_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("Unknowns / Gaps:")
        lines.extend(note_lines)
    else:
        insert_at = header_index + 1
        while insert_at < len(lines) and lines[insert_at].strip() not in _STRUCTURED_SECTION_HEADERS:
            insert_at += 1
        lines[insert_at:insert_at] = note_lines

    return "\n".join(lines).strip()


def _format_structured_answer(answer: str) -> str:
    """Insert clear paragraph breaks before structured section headers."""
    lines = answer.splitlines()
    formatted_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in _STRUCTURED_SECTION_HEADERS and formatted_lines:
            if formatted_lines[-1] != "":
                formatted_lines.append("")
        formatted_lines.append(line)

    formatted = "\n".join(formatted_lines)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)
    return formatted.strip()


def _build_chunk_dict(chunk: RetrievedChunk) -> dict:
    """Serialize a retrieved chunk for evaluation and inspection."""
    chunk_dict = {
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "source": chunk.metadata.get("source", "unknown"),
        "title": chunk.metadata.get("title", ""),
        "score": chunk.score,
        "rank": chunk.rank + 1,
        "metadata": dict(chunk.metadata),
    }
    chunk_dict["citation_label"] = build_citation_label(chunk_dict)
    chunk_dict["cited_in_answer"] = False
    return chunk_dict


def _log_used_chunks(question: str, chunk_dicts: list[dict]) -> None:
    """Log the final chunks that were passed to the generation step."""
    logger.info("Final retrieved chunks for question=%r: %d", question, len(chunk_dicts))
    for chunk in chunk_dicts:
        logger.info(
            "  rank=%s score=%.4f doc_id=%s cited_in_answer=%s title=%r",
            chunk.get("rank", "?"),
            chunk.get("score", 0.0),
            chunk.get("doc_id", "unknown"),
            chunk.get("cited_in_answer", False),
            chunk.get("title", ""),
        )


# ---------------------------------------------------------------------------
# Templated-chain helpers (Phase 2)
#
# These helpers make the Phase-2 pipeline deterministic about *which* chunks
# actually feed the fact bundle. The two responsibilities kept here are:
#
#   1. ``_auto_detect_cves_from_chunks`` — CrossSourceCompare queries that
#      don't name explicit CVEs (e.g. "which recent Citrix CVEs appear in
#      both NVD and CISA KEV") need entity pivots derived from the retrieved
#      context before the fact bundle can be grouped.
#
#   2. ``_augment_chunks_for_cves`` — entity-aware recall fallback. The
#      hybrid ranker still loses the primary NVD / KEV chunk for some
#      exact-CVE questions (see L1 smoke test notes and Sprint-3 retrieval
#      hardening). Rather than silently returning an empty bundle, we ask
#      Chroma for the NVD + KEV chunks of every primary CVE directly,
#      bounded per CVE so context doesn't blow up. This is conservative:
#      we augment only when a primary entity's top-2 chunks are missing.
#
# Both mirror the logic in ``tests/run_l1_smoke.py`` — the idea is to keep
# one canonical implementation in the chain and import-or-call it from the
# smoke test if we want to consolidate later.
# ---------------------------------------------------------------------------


_AUGMENT_PER_CVE_LIMIT = 2
_AUGMENT_RELEVANT_SOURCES = ("nvd", "cisa_kev")
# Hard ceiling on the context size after entity-aware augmentation. Without
# it, CrossSourceCompare queries with many auto-detected CVEs ballooned to
# 24 contexts, which measurably hurt context_precision and diluted the
# generation prompt.
_AUGMENT_MAX_TOTAL_CHUNKS = 10


def _auto_detect_cves_from_chunks(chunks: list[dict], top_n: int = 10) -> list[str]:
    """Return the most-mentioned CVE IDs across retrieved chunks."""
    counts: Counter[str] = Counter()
    for chunk in chunks:
        text = f"{chunk.get('title', '') or ''} {chunk.get('content', '') or ''}"
        for cve in extract_cve_ids(text):
            counts[cve] += 1
        meta = chunk.get("metadata") or {}
        raw = meta.get("cve_ids")
        if isinstance(raw, str):
            for cve in raw.split(","):
                cve = cve.strip().upper()
                if cve:
                    counts[cve] += 1
    return [cve for cve, _ in counts.most_common(top_n)]


def _augment_chunks_for_cves(
    retriever: HybridRetriever,
    chunks: list[dict],
    cve_ids: list[str],
) -> tuple[list[dict], list[str]]:
    """
    Ensure at most ``_AUGMENT_PER_CVE_LIMIT`` primary-source chunks per CVE.

    Returns ``(augmented_chunks, cves_augmented)`` where ``cves_augmented``
    lists the CVEs for which we had to pull extra chunks from Chroma.
    """
    if not cve_ids or getattr(retriever, "collection", None) is None:
        return chunks, []

    existing_by_cve: Counter[str] = Counter()
    existing_ids: set[str] = {c.get("doc_id", "") for c in chunks if c.get("doc_id")}

    def _doc_id_matches(doc_id: str, cve: str) -> bool:
        return cve.upper() in (doc_id or "").upper()

    for chunk in chunks:
        source = (chunk.get("metadata") or {}).get("source") or chunk.get("source") or ""
        if source not in _AUGMENT_RELEVANT_SOURCES:
            continue
        doc_id = chunk.get("doc_id", "")
        for cve in cve_ids:
            if _doc_id_matches(doc_id, cve):
                existing_by_cve[cve.upper()] += 1

    augmented = list(chunks)
    augmented_cves: list[str] = []

    for cve in cve_ids:
        if len(augmented) >= _AUGMENT_MAX_TOTAL_CHUNKS:
            logger.info(
                "Entity-aware augmentation stopped at %d chunks (cap); remaining CVEs skipped.",
                len(augmented),
            )
            break
        cve_u = cve.upper()
        if existing_by_cve.get(cve_u, 0) >= _AUGMENT_PER_CVE_LIMIT:
            continue
        try:
            res = retriever.collection.get(
                where={"cve_ids": {"$eq": cve_u}}, limit=20
            )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Entity-aware augmentation failed for %s: %s", cve_u, exc)
            continue

        added = 0
        for doc_id, content, meta in zip(
            res.get("ids", []) or [],
            res.get("documents", []) or [],
            res.get("metadatas", []) or [],
        ):
            if len(augmented) >= _AUGMENT_MAX_TOTAL_CHUNKS:
                break
            if doc_id in existing_ids:
                continue
            meta_dict = dict(meta or {})
            source = meta_dict.get("source", "unknown")
            if source not in _AUGMENT_RELEVANT_SOURCES:
                continue
            title = meta_dict.get("title", "")
            chunk_dict = {
                "doc_id": doc_id,
                "content": content or "",
                "source": source,
                "title": title,
                "score": 0.0,
                "rank": len(augmented) + 1,
                "metadata": meta_dict,
            }
            chunk_dict["citation_label"] = build_citation_label(chunk_dict)
            chunk_dict["cited_in_answer"] = False
            augmented.append(chunk_dict)
            existing_ids.add(doc_id)
            added += 1
            if added >= _AUGMENT_PER_CVE_LIMIT:
                break

        if added:
            augmented_cves.append(cve_u)

    return augmented, augmented_cves


def _fact_bundle_to_dict(bundle) -> dict:
    """Serialize a :class:`FactBundle` for inclusion in ``RAGResponse``."""
    if bundle is None:
        return {}
    try:
        return asdict(bundle) if is_dataclass(bundle) else dict(bundle)
    except TypeError:
        # Fall back to a shallow best-effort dict if the bundle contains
        # anything non-serializable (it shouldn't — all fact dataclasses
        # nest only primitives/lists/dicts).
        return {"template": getattr(bundle, "template", None)}


@dataclass
class RAGResponse:
    """
    Complete RAG response with full provenance for evaluation.

    Stores everything needed for RAGAS evaluation:
    - query: the input question
    - answer: the LLM-generated response
    - contexts: the retrieved chunks (for context_precision, context_recall)
    - source_documents: full metadata for citation verification
    """
    query: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    source_documents: list[dict] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    retrieval_mode: str = "hybrid"
    prompt_context: str = ""
    abstention_reason: str | None = None
    grounding_warnings: list[str] = field(default_factory=list)
    retrieval_trace: dict = field(default_factory=dict)
    # Unnormalized LLM text, kept for citation-pipeline root-cause analysis.
    raw_llm_output: str = ""

    # Phase-2 templated-chain fields. Populated by ``RAGChain.query_templated``;
    # remain empty for the legacy ``query`` / ``query_baseline`` paths so
    # existing callers and tests keep working unchanged.
    template: str | None = None
    routing_decision: dict = field(default_factory=dict)
    fact_bundle: dict = field(default_factory=dict)
    l1_block: str = ""
    l2_output: str = ""


class RAGChain:
    """
    Main RAG orchestration chain.

    Explicit, step-by-step pipeline (not a black-box chain):
    1. Query received
    2. Hybrid retrieval (BM25 + Vector + RRF + Rerank)
    3. Context assembly with source IDs
    4. Prompt construction
    5. LLM generation via Ollama
    6. Response packaging with full provenance
    """

    def __init__(self, retrieval_mode: str = "hybrid"):
        config = load_config()
        llm_config = config["llm"]

        # Initialize LLM
        self.llm = ChatOllama(
            model=llm_config["model_name"],
            base_url=llm_config["base_url"],
            temperature=llm_config["temperature"],
            top_p=llm_config["top_p"],
            num_predict=llm_config["max_tokens"],
        )

        self.retrieval_mode = retrieval_mode
        self.retriever: HybridRetriever | None = None

        logger.info(f"RAGChain initialized (mode={retrieval_mode})")

    def query(self, question: str) -> RAGResponse:
        """
        Execute the full RAG pipeline for a given question.

        Returns a RAGResponse with full provenance for evaluation.
        """
        total_start = time.time()

        # --- Step 1: Retrieve ---
        if self.retriever is None:
            self.retriever = HybridRetriever(retrieval_mode=self.retrieval_mode)

        retrieval_start = time.time()
        chunks: list[RetrievedChunk] = self.retriever.retrieve(question)
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")

        # --- Step 2: Assemble context ---
        chunk_dicts = [_build_chunk_dict(chunk) for chunk in chunks]

        context_str = format_context(chunk_dicts)

        abstention_reason = _assess_context_support(question, chunk_dicts)
        if abstention_reason:
            total_time = (time.time() - total_start) * 1000
            logger.info("Abstaining before generation: %s", abstention_reason)
            _log_used_chunks(question, chunk_dicts)
            return RAGResponse(
                query=question,
                answer=_build_abstention_answer(abstention_reason),
                contexts=[chunk.content for chunk in chunks],
                source_documents=chunk_dicts,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0,
                total_time_ms=total_time,
                retrieval_mode=self.retrieval_mode,
                prompt_context=context_str,
                abstention_reason=abstention_reason,
                retrieval_trace=dict(getattr(self.retriever, "last_trace", {})),
            )

        # --- Step 3: Build prompt ---
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"--- Retrieved Context ---\n{context_str}\n--- End of Context ---\n\n{QUERY_TEMPLATE.format(query=question)}"),
        ]

        # --- Step 4: Generate ---
        generation_start = time.time()
        response = self.llm.invoke(messages)
        generation_time = (time.time() - generation_start) * 1000
        raw_llm_output = response.content
        answer = _normalize_response_citations(raw_llm_output, chunk_dicts)
        answer = _enforce_summary_grounding(answer)
        answer = _enforce_why_grounding(answer)
        answer = _enforce_evidence_grounding(answer)
        answer = _enforce_mitigation_grounding(answer)
        compliance = _analyze_citation_compliance(answer)
        grounding_warnings: list[str] = []
        if compliance["cited_lines"] == 0:
            abstention_reason = "The generated answer did not retain any verifiable citations after normalization."
            total_time = (time.time() - total_start) * 1000
            logger.info("Abstaining after generation: %s", abstention_reason)
            _log_used_chunks(question, chunk_dicts)
            return RAGResponse(
                query=question,
                answer=_build_abstention_answer(abstention_reason),
                contexts=[chunk.content for chunk in chunks],
                source_documents=chunk_dicts,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                retrieval_mode=self.retrieval_mode,
                prompt_context=context_str,
                abstention_reason=abstention_reason,
                retrieval_trace=dict(getattr(self.retriever, "last_trace", {})),
                raw_llm_output=raw_llm_output,
            )

        if not compliance["has_evidence_section"] or compliance["evidence_cited_lines"] == 0:
            grounding_warnings.append("The answer lacks a clearly source-grounded evidence section.")
        if compliance["uncited_lines"] > 0:
            grounding_warnings.append(
                f"{compliance['uncited_lines']} response line(s) are not directly backed by surviving citations."
            )

        answer = _append_grounding_notes(answer, grounding_warnings)
        answer = _format_structured_answer(answer)
        cited_labels = _extract_citation_labels(answer)
        for chunk_dict in chunk_dicts:
            chunk_dict["cited_in_answer"] = chunk_dict["citation_label"] in cited_labels

        total_time = (time.time() - total_start) * 1000
        logger.info(f"Generated response in {generation_time:.0f}ms (total: {total_time:.0f}ms)")
        _log_used_chunks(question, chunk_dicts)

        # --- Step 5: Package response ---
        return RAGResponse(
            query=question,
            answer=answer,
            contexts=[chunk.content for chunk in chunks],
            source_documents=chunk_dicts,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieval_mode=self.retrieval_mode,
            prompt_context=context_str,
            grounding_warnings=grounding_warnings,
            retrieval_trace=dict(getattr(self.retriever, "last_trace", {})),
            raw_llm_output=raw_llm_output,
        )

    def query_templated(self, question: str) -> RAGResponse:
        """
        Phase-2 pipeline: router + fact bundle + task-specific template.

        Pipeline stages (see docs/triage-template-spec.md §§ 3, 5, 7):
          1. Deterministic router → ``RoutingDecision`` (template, primary
             entities, threat-context kind).
          2. Hybrid retrieval — unchanged from legacy ``query``.
          3. Entity-aware recall augmentation for CVE-centric templates:
             top-2 NVD/KEV chunks per primary CVE are force-fetched from
             Chroma if missing. Guards against the known recall gap for
             exact-CVE queries (Sprint-3 retrieval hardening).
          4. CrossSourceCompare entity fallback: if the query names no
             CVEs explicitly, derive primary entities from chunk mentions.
          5. Fact Bundle construction + severity annotation (L1).
          6. Task-specific prompt via ``templates.build_prompt`` (L2).
          7. LLM generation, citation normalization, lightweight grounding
             analysis.
          8. Final answer = ``{L1-block}\\n\\n{L2-output}`` — the L1 block
             is deterministic markdown that the analyst can trust verbatim;
             the L2 output is the LLM's grounded, cited narrative.

        ``query`` is kept intact as the legacy Phase-1 entry point so
        existing regression tests and SRQ1 baselines stay reproducible
        while the Field-Coverage evaluator validates the templated path.
        """
        total_start = time.time()

        # --- Step 1: Route ---
        decision: RoutingDecision = classify_query(question)
        logger.info(
            "Router → template=%s rule=%d primary=%s kind=%s",
            decision.template,
            decision.rule_matched,
            decision.primary_entities,
            decision.threat_context_kind,
        )

        # --- Step 2: Retrieve ---
        if self.retriever is None:
            self.retriever = HybridRetriever(retrieval_mode=self.retrieval_mode)

        retrieval_start = time.time()
        chunks: list[RetrievedChunk] = self.retriever.retrieve(question)
        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info("Retrieved %d chunks in %.0fms", len(chunks), retrieval_time)

        chunk_dicts = [_build_chunk_dict(chunk) for chunk in chunks]

        # --- Step 3: Entity-aware recall augmentation ---
        primary_entities = list(decision.primary_entities)

        # Step 4 (before augmentation): CrossSourceCompare without CVEs →
        # derive from chunks. Must happen first because augmentation reads
        # the resolved primary_entities.
        if decision.template == "CrossSourceCompare" and not primary_entities:
            primary_entities = _auto_detect_cves_from_chunks(chunk_dicts)
            logger.info(
                "CrossSourceCompare auto-detected entities: %s", primary_entities
            )

        augmented_cves: list[str] = []
        if decision.template in ("VulnTriage", "CrossSourceCompare") and primary_entities:
            cve_entities = [e for e in primary_entities if e.upper().startswith("CVE-")]
            if cve_entities:
                chunk_dicts, augmented_cves = _augment_chunks_for_cves(
                    self.retriever, chunk_dicts, cve_entities
                )
                if augmented_cves:
                    logger.info(
                        "Entity-aware augmentation pulled chunks for: %s",
                        augmented_cves,
                    )

        # --- Step 5: Context support check + abstention ---
        context_str = format_context(chunk_dicts)
        abstention_reason = _assess_context_support(question, chunk_dicts)
        if abstention_reason:
            total_time = (time.time() - total_start) * 1000
            logger.info("Abstaining before generation: %s", abstention_reason)
            _log_used_chunks(question, chunk_dicts)
            return RAGResponse(
                query=question,
                answer=_build_abstention_answer(abstention_reason),
                contexts=[c.get("content", "") for c in chunk_dicts],
                source_documents=chunk_dicts,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0,
                total_time_ms=total_time,
                retrieval_mode=self.retrieval_mode,
                prompt_context=context_str,
                abstention_reason=abstention_reason,
                retrieval_trace=dict(getattr(self.retriever, "last_trace", {})),
                template=decision.template,
                routing_decision=decision.as_trace_dict(),
            )

        # --- Step 6: Build fact bundle (L1) ---
        bundle = build_fact_bundle(
            decision.template,
            primary_entities,
            chunk_dicts,
            threat_context_kind=decision.threat_context_kind,
        )

        # Severity annotation so the L1 renderer and the evaluator both see
        # triage signals attached. VulnTriage and CrossSourceCompare both
        # carry VulnTriageFacts entities; ThreatContext has none.
        if decision.template == "VulnTriage":
            for facts in bundle.vuln_triage:
                annotate_vuln_triage_bundle(facts)
        elif decision.template == "CrossSourceCompare" and bundle.cross_source:
            for facts in bundle.cross_source.entities:
                annotate_vuln_triage_bundle(facts)

        l1_block = render_l1(bundle)

        # --- Step 7: Build template prompt + generate (L2) ---
        messages_dict = build_template_prompt(bundle, question, chunk_dicts)
        messages = [
            SystemMessage(content=m["content"]) if m["role"] == "system"
            else HumanMessage(content=m["content"])
            for m in messages_dict
        ]

        generation_start = time.time()
        response = self.llm.invoke(messages)
        generation_time = (time.time() - generation_start) * 1000

        # Citation normalization on the L2 output only. The L1 block is
        # deterministic markdown assembled from chunk metadata — it has no
        # LLM-authored citations to normalize. Patch 1 (Step 12) added
        # fuzzy alias resolution so the normalizer now recovers bare CVE
        # IDs, doc-id suffixes, and title-token matches instead of
        # stripping them.
        raw_llm_output = response.content
        l2_output = _normalize_response_citations(raw_llm_output, chunk_dicts)
        l2_output = re.sub(r"\n{3,}", "\n\n", l2_output).strip()

        # --- Step 8: Compose final answer ---
        final_answer = "\n\n".join(part for part in (l1_block, l2_output) if part)

        # Mark which chunks were actually cited.
        cited_labels = _extract_citation_labels(l2_output)
        for chunk_dict in chunk_dicts:
            chunk_dict["cited_in_answer"] = chunk_dict["citation_label"] in cited_labels

        # Lightweight grounding signals. The Phase-2 sections use bold
        # markdown headers (e.g. **Evidence**) rather than the Phase-1
        # plain-text "Evidence:" header — so ``_analyze_citation_compliance``
        # under-counts here. We report the simpler numbers for now and
        # defer full Phase-2 grounding analysis to the Field-Coverage
        # Evaluator (Sprint 3, Step 8).
        cited_line_count = sum(
            1 for line in l2_output.splitlines()
            if _CITATION_BLOCK_RE.search(line)
        )
        grounding_warnings: list[str] = []
        if cited_line_count == 0 and l2_output:
            grounding_warnings.append(
                "L2 output contains no verifiable citations after normalization."
            )

        total_time = (time.time() - total_start) * 1000
        logger.info(
            "Templated generation in %.0fms (total: %.0fms, cited_lines=%d)",
            generation_time,
            total_time,
            cited_line_count,
        )
        _log_used_chunks(question, chunk_dicts)

        trace = dict(getattr(self.retriever, "last_trace", {}))
        if augmented_cves:
            trace["entity_aware_augmented_cves"] = augmented_cves

        return RAGResponse(
            query=question,
            answer=final_answer,
            contexts=[c.get("content", "") for c in chunk_dicts],
            source_documents=chunk_dicts,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieval_mode=self.retrieval_mode,
            prompt_context=context_str,
            grounding_warnings=grounding_warnings,
            retrieval_trace=trace,
            raw_llm_output=raw_llm_output,
            template=decision.template,
            routing_decision=decision.as_trace_dict(),
            fact_bundle=_fact_bundle_to_dict(bundle),
            l1_block=l1_block,
            l2_output=l2_output,
        )

    def query_baseline(self, question: str, snapshot_date: str | None = None) -> RAGResponse:
        """
        Query LLM WITHOUT retrieval augmentation (baseline for SRQ1 comparison).
        """
        total_start = time.time()

        messages = [
            SystemMessage(content=message["content"])
            if message["role"] == "system"
            else HumanMessage(content=message["content"])
            for message in build_baseline_prompt(question, snapshot_date=snapshot_date)
        ]

        generation_start = time.time()
        response = self.llm.invoke(messages)
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - total_start) * 1000
        answer = _format_structured_answer(response.content)

        return RAGResponse(
            query=question,
            answer=answer,
            contexts=[],
            source_documents=[],
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieval_mode="baseline_no_retrieval",
            retrieval_trace={},
        )
