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
from dataclasses import dataclass, field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from ..utils.config import load_config
from .prompts import (
    SYSTEM_PROMPT,
    format_context,
    QUERY_TEMPLATE,
    build_citation_label,
    build_baseline_prompt,
)

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


def _build_citation_alias_map(source_documents: list[dict]) -> dict[str, str]:
    """Map allowed citation aliases back to their canonical citation label."""
    alias_map: dict[str, str] = {}
    ambiguous_aliases: set[str] = set()

    for doc in source_documents:
        canonical = doc.get("citation_label", "").strip()
        if not canonical:
            continue

        for alias in (canonical, doc.get("doc_id", ""), doc.get("title", "")):
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

    return alias_map


def _normalize_response_citations(answer: str, source_documents: list[dict]) -> str:
    """
    Normalize citation blocks to the canonical [Source: <citation_label>] form.

    Invalid or ambiguous citation aliases are removed rather than preserved.
    """
    alias_map = _build_citation_alias_map(source_documents)
    if not alias_map:
        return answer

    def replace(match: re.Match) -> str:
        raw_value = match.group(1).strip()
        candidates = [raw_value]
        if ";" in raw_value:
            candidates = [part.strip() for part in raw_value.split(";") if part.strip()]

        canonical_labels: list[str] = []
        for candidate in candidates:
            canonical = alias_map.get(_normalize_alias(candidate))
            if not canonical:
                return ""
            if canonical not in canonical_labels:
                canonical_labels.append(canonical)

        return " ".join(f"[Source: {label}]" for label in canonical_labels)

    normalized = _CITATION_BLOCK_RE.sub(replace, answer)
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
        answer = _normalize_response_citations(response.content, chunk_dicts)
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
