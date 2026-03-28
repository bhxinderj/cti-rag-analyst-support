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
1. ONLY use information from the provided context to answer. Do NOT use prior knowledge.
2. ALWAYS cite your sources using [Source: <doc_id>] format after each claim.
3. If the context does not contain enough information to answer, explicitly state what is missing.
4. Structure your response clearly with relevant sections (e.g., Summary, Technical Details, Impact, Recommendations).
5. For vulnerability queries, include severity, affected products, and known exploitation status when available.
6. Be precise with technical details - CVE IDs, CVSS scores, ATT&CK technique IDs must be exact."""


CONTEXT_TEMPLATE = """--- Retrieved Context ---
{context}
--- End of Context ---"""


QUERY_TEMPLATE = """Based on the retrieved context above, answer the following question:

{query}

Remember to cite sources using [Source: <doc_id>] format."""


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

        context_parts.append(
            f"[{i}] Source ID: {doc_id}\n"
            f"    Source Type: {source}\n"
            f"    Title: {title}\n"
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
