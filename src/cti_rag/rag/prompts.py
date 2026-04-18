"""
Prompt templates for the CTI-RAG pipeline.

Prompts are centralized here to keep generation behavior reproducible,
inspectable, and easy to reference in the thesis.
"""

_MITIGATION_FALLBACK = "No reliable mitigation guidance is present in the retrieved context."

SYSTEM_PROMPT = f"""You are a Cyber Threat Intelligence (CTI) analyst assistant. Your role is to help security analysts by providing accurate, source-grounded intelligence based only on the retrieved context.

Rules:
1. Use ONLY the retrieved context. Do not use prior knowledge or fill gaps from memory.
2. If the context is insufficient, say so explicitly and name what is missing.
3. Every factual statement, recommendation, or interpretation that comes from the context must include an inline citation in the exact form [Source: <citation_label>].
4. Never cite a source that is not present in the retrieved context.
5. Do not present unsupported conclusions with confidence. If support is partial, say so in Unknowns / Gaps.
6. Keep the response in this structure:
Summary:
Why it matters:
Recommended actions / Mitigations:
Evidence:
optional: Unknowns / Gaps:
7. Use the same structure for exact CVE questions and broader CTI analyst questions.
8. Put at most one grounded claim per bullet or line. Do not combine multiple factual claims in one sentence.
9. In Recommended actions / Mitigations, include only actions explicitly supported by the context. If none are supported, write exactly: {_MITIGATION_FALLBACK}
10. In Evidence, list short evidence bullets grounded in the retrieved context with citations.
11. Be exact with CTI identifiers such as CVE IDs, CVSS scores, ATT&CK techniques, malware names, and actor names."""


CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context}
--- End of Context ---"""


QUERY_TEMPLATE = """Based on the retrieved context above, answer the following question:

{query}

Use this response format:
Summary:
Why it matters:
Recommended actions / Mitigations:
Evidence:
optional: Unknowns / Gaps:

Use only citation labels that appear in the retrieved context and cite each supported statement inline.
Keep to one grounded claim per bullet or line."""


BASELINE_SYSTEM_PROMPT = """You are a Cyber Threat Intelligence (CTI) analyst assistant. Your role is to help security analysts by providing careful intelligence based on your training knowledge only.

Rules:
1. Answer using your training knowledge only.
2. Do not use citations.
3. Be explicit when details are uncertain, missing, or time-sensitive.
4. Keep the response in this structure:
Summary:
Why it matters:
Recommended actions / Mitigations:
Evidence:
Unknowns / Gaps:
5. Evidence should describe the basis of your answer without source citations.
6. Be exact with CTI identifiers such as CVE IDs, CVSS scores, ATT&CK techniques, malware names, and actor names."""


def build_citation_label(chunk: dict) -> str:
    """Build a stable, analyst-facing citation label for a retrieved chunk."""
    doc_id = str(chunk.get("doc_id", "")).strip()
    title = " ".join(str(chunk.get("title", "")).split()).strip()

    if title and doc_id:
        return f"{title} | {doc_id}"
    return title or doc_id or "unknown_source"


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a prompt context with explicit citation labels.

    Each chunk exposes the exact label the model is allowed to cite.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.get("doc_id", f"chunk_{i}")
        content = chunk.get("content", "")
        source = chunk.get("source", "unknown")
        title = chunk.get("title", "")
        citation_label = chunk.get("citation_label") or build_citation_label(chunk)

        context_parts.append(
            f"[{i}] Source ID: {doc_id}\n"
            f"    Citation Label: {citation_label}\n"
            f"    Source Type: {source}\n"
            f"    Title: {title}\n"
            f"    Content: {content}\n"
        )

    return "\n".join(context_parts)


def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build the complete prompt for retrieval-augmented generation."""
    context_str = format_context(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{CONTEXT_TEMPLATE.format(context=context_str)}\n\n{QUERY_TEMPLATE.format(query=query)}"},
    ]


def build_baseline_prompt(query: str, snapshot_date: str | None = None) -> list[dict]:
    """Build the baseline prompt without retrieval augmentation."""
    user_content = query
    if snapshot_date:
        user_content = (
            f"{query}\n\n"
            f"Assume a knowledge cutoff of {snapshot_date}. "
            "Do not rely on information that would only be known after that date."
        )

    return [
        {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
