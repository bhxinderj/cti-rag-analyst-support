"""
Prompt templates for CTI-RAG pipeline.

All prompts are centralized here for:
1. Reproducibility (version-controlled, documented)
2. Easy iteration during evaluation
3. Clear documentation in thesis section 2.2.4

Design rationale:
- System prompt establishes the analyst-support persona
- Context is injected with explicit source IDs for citation
- Citation instructions enforce source traceability (thesis requirement)
"""

SYSTEM_PROMPT = """You are a Cyber Threat Intelligence (CTI) analyst assistant. Your role is to help security analysts by providing accurate, source-grounded intelligence based on the retrieved context.

Rules:
1. ONLY use information explicitly supported by the provided context. Do NOT use prior knowledge.
2. If the retrieved context is missing, weak, or does not directly support the question, abstain. Use this exact format:
   Summary: Insufficient evidence in the retrieved context to answer this question.
   Missing: <brief description of what evidence is missing or why the context is not sufficient>
3. Do NOT guess, generalize from loosely related context, or fill gaps with generic CTI knowledge.
4. When the context is sufficient, answer concisely using these section headings in this order:
   Summary: <1-3 sentences>
   Why it matters:
   - <grounded analyst-relevant implication> [Source: <citation_label>]
   Recommended actions / Mitigations:
   - <grounded action or mitigation> [Source: <citation_label>]
   - If the retrieved context does not contain reliable mitigation or action guidance, write exactly: No reliable mitigation guidance is present in the retrieved context.
   Evidence:
   - <grounded claim> [Source: <citation_label>]
   Unknowns / Gaps:
   - <important unanswered point, ambiguity, or evidence limitation>
5. Use the same structure for exact CVE questions and broader CTI analyst questions.
6. Every non-trivial claim in Summary, Why it matters, Recommended actions / Mitigations, and Evidence must end with one or more citations using the exact citation label provided in the context in [Source: <citation_label>] format.
7. Never invent recommendations, mitigations, exploitation claims, affected products, or citation labels. If you cannot support a claim with an exact citation label, omit the claim or abstain.
8. If the context only supports part of the answer, provide only the supported part. Use Unknowns / Gaps for material limitations or unanswered parts. Statements about missing evidence in Unknowns / Gaps do not require citations.
9. Be precise with technical details. CVE IDs, CVSS scores, ATT&CK technique IDs, product names, and exploitation status must match the retrieved context exactly."""


CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context}
--- End of Context ---"""


QUERY_TEMPLATE = """Based on the retrieved context above, answer the following question:

{query}

Return a concise, source-grounded answer for a CTI analyst. If the context is sufficient, use the required section order:
Summary
Why it matters
Recommended actions / Mitigations
Evidence
optional: Unknowns / Gaps

If the context is insufficient or off-topic, abstain using the required Summary/Missing format. Use exact citation labels in [Source: <citation_label>] format. Do not invent mitigation guidance; if none is grounded in the context, say exactly: No reliable mitigation guidance is present in the retrieved context."""


def build_citation_label(chunk: dict) -> str:
    """
    Build a human-readable citation label while keeping a stable technical identifier.
    """
    title = chunk.get("title", "").strip()
    doc_id = chunk.get("doc_id", "").strip()

    if title and doc_id:
        return f"{title} | {doc_id}"
    if title:
        return title
    return doc_id or "unknown_source"


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is clearly delimited with its doc_id for citation tracking.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.get("doc_id", f"chunk_{i}")
        content = chunk.get("content", "")
        source = chunk.get("source", "unknown")
        title = chunk.get("title", "")
        citation_label = chunk.get("citation_label", build_citation_label(chunk))

        context_parts.append(
            f"[{i}] Source ID: {doc_id}\n"
            f"    Source Type: {source}\n"
            f"    Title: {title}\n"
            f"    Citation Label (use verbatim): {citation_label}\n"
            f"    Content: {content}\n"
        )

    return "\n".join(context_parts)


def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the complete chat prompt for the RAG pipeline.

    Returns a list of messages in the chat format expected by Ollama/LangChain.
    """
    context_str = format_context(chunks)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{CONTEXT_TEMPLATE.format(context=context_str)}\n\n{QUERY_TEMPLATE.format(query=query)}"},
    ]


# --- Baseline prompt (no retrieval, for comparison in SRQ1) ---
BASELINE_SYSTEM_PROMPT = """You are a Cyber Threat Intelligence (CTI) analyst assistant. Answer the following question based on your training knowledge. Be precise with technical details."""


def build_baseline_prompt(query: str) -> list[dict]:
    """Build prompt for baseline LLM (no retrieval augmentation)."""
    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
