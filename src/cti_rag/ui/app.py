"""
CTI-RAG Analyst Workbench — Streamlit Demo UI.

A lightweight demonstration interface for the CTI-RAG pipeline.
Connects to the FastAPI backend at the configured API URL.

    streamlit run src/cti_rag/ui/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("CTI_RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CTI-RAG Analyst Workbench",
    page_icon=":shield:",
    layout="wide",
)


def check_backend_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    return None


def query_rag(question: str, mode: str) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/api/query",
            json={"question": question, "mode": mode},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API server. Is it running?")
        return None
    except requests.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text[:300]}")
        return None
    except requests.Timeout:
        st.error("Request timed out. The LLM may be overloaded.")
        return None


# --- Header ---
st.title(":shield: CTI-RAG Analyst Workbench")
st.caption("Retrieval-Augmented Generation for Cyber Threat Intelligence")

# --- Sidebar: Status & Settings ---
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "bm25", "vector"],
        index=0,
        help="hybrid = BM25 + Vector + RRF fusion with cross-encoder reranking",
    )

    st.divider()
    st.header("Backend Status")
    health = check_backend_health()
    if health is None:
        st.error("API server unreachable")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Ollama", "OK" if health["ollama_reachable"] else "DOWN")
        col2.metric("Index", "OK" if health["index_loaded"] else "MISSING")
        st.caption(f"Setup: {health['active_setup']}")

def _render_sources(sources: list[dict], meta: dict):
    """Render source documents and metadata below the answer."""
    cited = [s for s in sources if s.get("cited_in_answer")]
    uncited = [s for s in sources if not s.get("cited_in_answer")]

    timing = (
        f"Retrieval: {meta.get('retrieval_time_ms', 0):.0f}ms · "
        f"Generation: {meta.get('generation_time_ms', 0):.0f}ms · "
        f"Total: {meta.get('total_time_ms', 0):.0f}ms"
    )
    st.caption(timing)

    if meta.get("grounding_warnings"):
        for warning in meta["grounding_warnings"]:
            st.warning(warning, icon=":warning:")

    if cited:
        with st.expander(f"Cited Sources ({len(cited)})", expanded=True):
            for src in cited:
                st.markdown(
                    f"**[{src['doc_id']}]** {src['title'][:100]}  \n"
                    f"`{src['source']}` · score: {src['score']:.4f} · rank: {src['rank']}"
                )
    if uncited:
        with st.expander(f"Retrieved Context — not cited ({len(uncited)})"):
            for src in uncited:
                st.markdown(
                    f"**[{src['doc_id']}]** {src['title'][:100]}  \n"
                    f"`{src['source']}` · score: {src['score']:.4f} · rank: {src['rank']}"
                )


# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"], msg.get("meta", {}))

# --- Chat Input ---
if question := st.chat_input("Ask a CTI question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = query_rag(question, mode)

        if result:
            st.markdown(result["answer"])

            meta = {
                "retrieval_time_ms": result["retrieval_time_ms"],
                "generation_time_ms": result["generation_time_ms"],
                "total_time_ms": result["total_time_ms"],
                "grounding_warnings": result.get("grounding_warnings", []),
            }
            _render_sources(result["sources"], meta)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "meta": meta,
            })
        else:
            fallback = "Failed to get a response from the backend."
            st.error(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
