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
    BASELINE_SYSTEM_PROMPT,
    build_citation_label,
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

        # Initialize retriever
        self.retriever = HybridRetriever(retrieval_mode=retrieval_mode)
        self.retrieval_mode = retrieval_mode

        logger.info(f"RAGChain initialized (mode={retrieval_mode})")

    def query(self, question: str) -> RAGResponse:
        """
        Execute the full RAG pipeline for a given question.

        Returns a RAGResponse with full provenance for evaluation.
        """
        total_start = time.time()

        # --- Step 1: Retrieve ---
        retrieval_start = time.time()
        chunks: list[RetrievedChunk] = self.retriever.retrieve(question)
        retrieval_time = (time.time() - retrieval_start) * 1000

        logger.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")

        # --- Step 2: Assemble context ---
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "source": chunk.metadata.get("source", "unknown"),
                "title": chunk.metadata.get("title", ""),
                "score": chunk.score,
            }
            chunk_dict["citation_label"] = build_citation_label(chunk_dict)
            chunk_dicts.append(chunk_dict)

        context_str = format_context(chunk_dicts)

        abstention_reason = _assess_context_support(question, chunk_dicts)
        if abstention_reason:
            total_time = (time.time() - total_start) * 1000
            logger.info("Abstaining before generation: %s", abstention_reason)
            return RAGResponse(
                query=question,
                answer=_build_abstention_answer(abstention_reason),
                contexts=[chunk.content for chunk in chunks],
                source_documents=chunk_dicts,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0,
                total_time_ms=total_time,
                retrieval_mode=self.retrieval_mode,
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

        total_time = (time.time() - total_start) * 1000
        logger.info(f"Generated response in {generation_time:.0f}ms (total: {total_time:.0f}ms)")

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
        )

    def query_baseline(self, question: str) -> RAGResponse:
        """
        Query LLM WITHOUT retrieval augmentation (baseline for SRQ1 comparison).
        """
        total_start = time.time()

        messages = [
            SystemMessage(content=BASELINE_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        generation_start = time.time()
        response = self.llm.invoke(messages)
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - total_start) * 1000

        return RAGResponse(
            query=question,
            answer=response.content,
            contexts=[],
            source_documents=[],
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            retrieval_mode="baseline_no_retrieval",
        )
