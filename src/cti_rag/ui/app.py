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

# --- Page Config ---
st.set_page_config(
    page_title="CTI-RAG Analyst Workbench",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* --- Header bar --- */
    .cti-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3b82f6;
    }
    .cti-header h1 {
        color: #f1f5f9;
        font-size: 1.6rem;
        margin: 0 0 0.25rem 0;
        font-weight: 700;
    }
    .cti-header p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0;
    }

    /* --- Source badges --- */
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-nvd { background: #1e3a5f; color: #60a5fa; }
    .badge-cisa_kev { background: #5c2d0e; color: #fb923c; }
    .badge-cisa_advisory { background: #5c2d0e; color: #fbbf24; }
    .badge-misp { background: #14532d; color: #4ade80; }
    .badge-unknown { background: #374151; color: #9ca3af; }

    /* --- Source card --- */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .source-card .doc-id {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.85rem;
    }
    .source-card .doc-title {
        color: #475569;
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .source-card .doc-meta {
        color: #94a3b8;
        font-size: 0.72rem;
        margin-top: 4px;
    }

    /* --- Grounding indicator --- */
    .grounding-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .grounding-ok { background: #14532d; color: #4ade80; }
    .grounding-warn { background: #713f12; color: #fbbf24; }
    .grounding-fail { background: #7f1d1d; color: #fca5a5; }

    /* --- Timing bar --- */
    .timing-bar {
        display: flex;
        gap: 1.2rem;
        padding: 0.4rem 0;
        font-size: 0.75rem;
        color: #64748b;
    }
    .timing-bar span {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    /* --- Welcome screen --- */
    .welcome-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 700px;
    }
    .welcome-box h3 {
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .welcome-box p {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* --- Pipeline sidebar info --- */
    .pipeline-step {
        font-size: 0.78rem;
        color: #cbd5e1;
        padding: 2px 0;
    }
    .pipeline-step strong {
        color: #f1f5f9;
    }

    /* --- Sidebar tweaks --- */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# --- Source badge mapping ---
_SOURCE_BADGE_CLASS = {
    "nvd": "badge-nvd",
    "cisa_kev": "badge-cisa_kev",
    "cisa_advisory": "badge-cisa_advisory",
    "misp": "badge-misp",
}

_SOURCE_DISPLAY_NAME = {
    "nvd": "NVD",
    "cisa_kev": "CISA KEV",
    "cisa_advisory": "CISA Advisory",
    "misp": "MISP",
}

EXAMPLE_QUERIES = [
    "What is CVE-2021-44228 and how has it been exploited?",
    "Which vulnerabilities are listed in the CISA KEV catalog for Apache products?",
    "What MITRE ATT&CK techniques are associated with Log4Shell?",
    "What are the most critical vulnerabilities with CVSS score above 9.0?",
]


# --- Helper functions ---

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
        st.error(f"API error: {exc.response.status_code} \u2014 {exc.response.text[:300]}")
        return None
    except requests.Timeout:
        st.error("Request timed out. The LLM may be overloaded.")
        return None


def _source_badge(source_type: str) -> str:
    badge_class = _SOURCE_BADGE_CLASS.get(source_type, "badge-unknown")
    display = _SOURCE_DISPLAY_NAME.get(source_type, source_type.upper())
    return f'<span class="source-badge {badge_class}">{display}</span>'


def _render_grounding_status(result: dict):
    if result.get("abstention_reason"):
        st.markdown(
            '<span class="grounding-pill grounding-fail">Abstained \u2014 insufficient evidence</span>',
            unsafe_allow_html=True,
        )
    elif result.get("grounding_warnings"):
        st.markdown(
            '<span class="grounding-pill grounding-warn">Partially grounded</span>',
            unsafe_allow_html=True,
        )
        for warning in result["grounding_warnings"]:
            st.caption(f"\u26a0\ufe0f {warning}")
    else:
        st.markdown(
            '<span class="grounding-pill grounding-ok">Fully grounded</span>',
            unsafe_allow_html=True,
        )


def _render_timing(meta: dict):
    st.markdown(
        f"""<div class="timing-bar">
            <span>\U0001f50d Retrieval: {meta.get('retrieval_time_ms', 0):.0f}ms</span>
            <span>\U0001f9e0 Generation: {meta.get('generation_time_ms', 0):.0f}ms</span>
            <span>\u23f1\ufe0f Total: {meta.get('total_time_ms', 0):.0f}ms</span>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_source_card(src: dict):
    badge = _source_badge(src.get("source", "unknown"))
    st.markdown(
        f"""<div class="source-card">
            <div>{badge} <span class="doc-id">{src['doc_id']}</span></div>
            <div class="doc-title">{src.get('title', '')[:120]}</div>
            <div class="doc-meta">Score: {src.get('score', 0):.4f} &middot; Rank: {src.get('rank', '-')}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_sources(sources: list[dict], meta: dict):
    """Render grounding status, timing, and source cards below the answer."""
    _render_grounding_status(meta)
    _render_timing(meta)

    cited = [s for s in sources if s.get("cited_in_answer")]
    uncited = [s for s in sources if not s.get("cited_in_answer")]

    if cited:
        with st.expander(f"\U0001f4ce Cited Sources ({len(cited)})", expanded=True):
            for src in cited:
                _render_source_card(src)
    if uncited:
        with st.expander(f"\U0001f4c2 Retrieved Context \u2014 not cited ({len(uncited)})"):
            for src in uncited:
                _render_source_card(src)


# --- Header ---
st.markdown("""
<div class="cti-header">
    <h1>\U0001f6e1\ufe0f CTI-RAG Analyst Workbench</h1>
    <p>Retrieval-Augmented Generation for Cyber Threat Intelligence</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    if st.button("New Chat", use_container_width=True, icon="\U0001f4ac"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### \u2699\ufe0f Settings")
    mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "bm25", "vector"],
        index=0,
        help="hybrid = BM25 + Vector + RRF fusion with cross-encoder reranking",
    )

    st.divider()

    st.markdown("### \U0001f4e1 Backend Status")
    health = check_backend_health()
    if health is None:
        st.error("API server unreachable")
    else:
        col1, col2 = st.columns(2)
        if health["ollama_reachable"]:
            col1.success("Ollama: OK", icon="\u2705")
        else:
            col1.error("Ollama: DOWN", icon="\u274c")
        if health["index_loaded"]:
            col2.success("Index: OK", icon="\u2705")
        else:
            col2.error("Index: MISSING", icon="\u274c")
        st.caption(f"Active setup: `{health['active_setup']}`")

    st.divider()

    st.markdown("### \U0001f527 Pipeline")
    st.markdown("""
<div>
    <div class="pipeline-step"><strong>1.</strong> BM25 lexical search (top-20)</div>
    <div class="pipeline-step"><strong>2.</strong> Vector similarity search (top-20)</div>
    <div class="pipeline-step"><strong>3.</strong> Reciprocal Rank Fusion (RRF k=60)</div>
    <div class="pipeline-step"><strong>4.</strong> Cross-encoder reranking (top-5)</div>
    <div class="pipeline-step"><strong>5.</strong> LLM generation with citation enforcement</div>
</div>
""", unsafe_allow_html=True)


# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Welcome Screen (only when no messages) ---
if not st.session_state.messages:
    st.markdown("""
<div class="welcome-box">
    <h3>Ask a question about cyber threats</h3>
    <p>This prototype retrieves evidence from NVD, CISA KEV, CISA Advisories, and MISP feeds,
    then generates a source-grounded answer with transparent citations.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUERIES):
        if cols[i % 2].button(example, key=f"example_{i}", use_container_width=True):
            st.session_state["_pending_query"] = example
            st.rerun()

# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="\U0001f6e1\ufe0f" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            _render_sources(msg["sources"], msg.get("meta", {}))

# --- Chat Input ---
pending = st.session_state.pop("_pending_query", None)
question = st.chat_input("Ask a CTI question...") or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="\U0001f6e1\ufe0f"):
        with st.spinner("Retrieving evidence and generating response..."):
            result = query_rag(question, mode)

        if result:
            st.markdown(result["answer"])

            meta = {
                "retrieval_time_ms": result["retrieval_time_ms"],
                "generation_time_ms": result["generation_time_ms"],
                "total_time_ms": result["total_time_ms"],
                "grounding_warnings": result.get("grounding_warnings", []),
                "abstention_reason": result.get("abstention_reason"),
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
