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


_CUSTOM_CSS = """
<style>
/* --- Global palette tweaks --- */
.stApp {
    background: radial-gradient(1200px 600px at 10% -10%, #14244a 0%, #0a1120 55%, #070d1a 100%);
}

/* --- Hero header --- */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #0ea5e9 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(148, 163, 184, 0.15);
}
.hero h1 {
    margin: 0;
    color: #f8fafc;
    font-size: 1.9rem;
    letter-spacing: -0.02em;
}
.hero p {
    margin: 4px 0 0 0;
    color: #cbd5e1;
    font-size: 0.95rem;
}

/* --- Sidebar polish --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a33 0%, #0a1120 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.1);
}
section[data-testid="stSidebar"] h2 {
    color: #38bdf8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

/* --- Chat messages --- */
[data-testid="stChatMessage"] {
    background: #111d38 !important;
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 12px;
    padding: 14px 18px !important;
    margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background: #142645 !important;
    border-left: 3px solid #38bdf8;
}
[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #22d3ee;
}

/* --- Inline "pill" tags for metadata --- */
.pill {
    display: inline-block;
    padding: 3px 10px;
    margin: 2px 4px 2px 0;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    background: rgba(56, 189, 248, 0.12);
    color: #7dd3fc;
    border: 1px solid rgba(56, 189, 248, 0.25);
}
.pill.template  { background: rgba(167, 139, 250, 0.12); color: #c4b5fd; border-color: rgba(167, 139, 250, 0.25); }
.pill.timing    { background: rgba(148, 163, 184, 0.10); color: #cbd5e1; border-color: rgba(148, 163, 184, 0.20); }
.pill.legacy    { background: rgba(250, 204, 21, 0.10); color: #fde68a; border-color: rgba(250, 204, 21, 0.25); }

/* --- Source cards --- */
.src-card {
    background: #0f1a33;
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.src-card.uncited { border-left-color: #475569; opacity: 0.85; }
.src-card .doc-id {
    color: #38bdf8;
    font-family: ui-monospace, "SF Mono", monospace;
    font-size: 0.82rem;
    font-weight: 600;
}
.src-card.uncited .doc-id { color: #94a3b8; }
.src-card .title {
    color: #e2e8f0;
    font-size: 0.92rem;
    margin-top: 2px;
}
.src-card .meta {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 4px;
    font-family: ui-monospace, "SF Mono", monospace;
}

/* --- Expander polish --- */
.streamlit-expanderHeader {
    background: #111d38 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(56, 189, 248, 0.12) !important;
}

/* --- Buttons / inputs --- */
.stChatInput textarea {
    background: #111d38 !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* --- Metric cards in sidebar --- */
[data-testid="stMetric"] {
    background: #0f1a33;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(56, 189, 248, 0.12);
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 1.15rem !important;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def check_backend_health() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.ConnectionError:
        return None
    return None


def query_rag(question: str, mode: str, templated: bool) -> dict | None:
    try:
        resp = requests.post(
            f"{API_URL}/api/query",
            json={"question": question, "mode": mode, "templated": templated},
            timeout=300,
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


# --- Hero header ---
st.markdown(
    """
    <div class="hero">
      <h1>🛡️ CTI-RAG Analyst Workbench</h1>
      <p>Retrieval-Augmented Generation for Cyber Threat Intelligence — Phase 2 Templated Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Status & Settings ---
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "bm25", "vector"],
        index=0,
        help="hybrid = BM25 + Vector + RRF fusion with cross-encoder reranking",
    )
    pipeline_choice = st.radio(
        "Pipeline",
        options=["Templated (Phase 2)", "Legacy (Phase 1)"],
        index=0,
        help=(
            "Templated = Router + Fact Bundle + L1/L2 task-specific templates "
            "(VulnTriage / ThreatContext / CrossSourceCompare). "
            "Legacy = single generic prompt."
        ),
    )
    templated = pipeline_choice.startswith("Templated")

    st.divider()
    st.header("Backend Status")
    health = check_backend_health()
    if health is None:
        st.error("API server unreachable")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Ollama", "OK" if health["ollama_reachable"] else "DOWN")
        col2.metric("Index", "OK" if health["index_loaded"] else "MISSING")
        st.caption(f"Setup: `{health['active_setup']}`")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def _pills(meta: dict) -> str:
    """Build the pill-badge row for pipeline/template/timing metadata."""
    pipeline = meta.get("pipeline", "legacy")
    template = meta.get("template")
    parts = []
    pipeline_cls = "pill" if pipeline == "templated" else "pill legacy"
    parts.append(f'<span class="{pipeline_cls}">pipeline · {pipeline}</span>')
    if template:
        parts.append(f'<span class="pill template">template · {template}</span>')
    parts.append(
        f'<span class="pill timing">retrieval {meta.get("retrieval_time_ms", 0):.0f}ms</span>'
    )
    parts.append(
        f'<span class="pill timing">generation {meta.get("generation_time_ms", 0):.0f}ms</span>'
    )
    parts.append(
        f'<span class="pill timing">total {meta.get("total_time_ms", 0):.0f}ms</span>'
    )
    return "<div>" + "".join(parts) + "</div>"


def _source_card_html(src: dict, cited: bool) -> str:
    cls = "src-card" if cited else "src-card uncited"
    title = (src.get("title") or "")[:120]
    return (
        f'<div class="{cls}">'
        f'<div class="doc-id">[{src["doc_id"]}]</div>'
        f'<div class="title">{title}</div>'
        f'<div class="meta">{src["source"]} · score {src["score"]:.4f} · rank {src["rank"]}</div>'
        f"</div>"
    )


def _render_sources(sources: list[dict], meta: dict):
    """Render source documents and metadata below the answer."""
    cited = [s for s in sources if s.get("cited_in_answer")]
    uncited = [s for s in sources if not s.get("cited_in_answer")]

    st.markdown(_pills(meta), unsafe_allow_html=True)

    if meta.get("grounding_warnings"):
        for warning in meta["grounding_warnings"]:
            st.warning(warning, icon=":warning:")

    routing = meta.get("routing_decision")
    if routing:
        with st.expander("Routing decision", expanded=False):
            st.json(routing)

    if cited:
        with st.expander(f"📎 Cited Sources ({len(cited)})", expanded=True):
            html = "".join(_source_card_html(s, cited=True) for s in cited)
            st.markdown(html, unsafe_allow_html=True)
    if uncited:
        with st.expander(f"📚 Retrieved context — not cited ({len(uncited)})"):
            html = "".join(_source_card_html(s, cited=False) for s in uncited)
            st.markdown(html, unsafe_allow_html=True)


# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"], msg.get("meta", {}))

# --- Chat Input ---
if question := st.chat_input("Ask a CTI question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        spinner_label = (
            "Routing → retrieving → generating (templated)…"
            if templated
            else "Retrieving and generating…"
        )
        with st.spinner(spinner_label):
            result = query_rag(question, mode, templated)

        if result:
            st.markdown(result["answer"])

            meta = {
                "retrieval_time_ms": result["retrieval_time_ms"],
                "generation_time_ms": result["generation_time_ms"],
                "total_time_ms": result["total_time_ms"],
                "grounding_warnings": result.get("grounding_warnings", []),
                "pipeline": result.get("pipeline", "legacy"),
                "template": result.get("template"),
                "routing_decision": result.get("routing_decision"),
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
