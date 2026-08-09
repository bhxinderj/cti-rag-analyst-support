"""
Unit tests for the Extended-Analysis chain (src/cti_rag/rag/extended.py).

The extended mode must be strictly additive: frozen modules are reused
import-only, the analyst-assist layer is appended to the unchanged
template system prompt, and the retrieval deepening is an
instance-level override.
"""

from __future__ import annotations

from unittest.mock import patch

from src.cti_rag.rag.extended import (
    _ANALYST_ASSIST_INSTRUCTIONS,
    _EXTENDED_RERANK_TOP_K,
    ExtendedChain,
)
from tests.test_chain_templated import (
    _FakeCollection,
    _FakeLLM,
    _FakeRetriever,
    _kev_rchunk,
    _nvd_rchunk,
)


def _make_extended(llm_response: str, chunks) -> tuple[ExtendedChain, _FakeLLM]:
    fake_llm = _FakeLLM(llm_response)

    def _fake_builder(llm_config):
        assert llm_config["provider"] == "openrouter"
        return "openrouter:test-model", fake_llm

    with patch("src.cti_rag.rag.extended.build_generation_llm", _fake_builder), patch(
        "src.cti_rag.rag.extended.HybridRetriever",
        lambda retrieval_mode: _FakeRetriever(chunks, collection=_FakeCollection({})),
    ):
        chain = ExtendedChain()
    return chain, fake_llm


def test_extended_chain_appends_assist_layer_and_keeps_template_prompt():
    llm_text = (
        "**Exploitation Context**\n"
        "- Session hijacking [Source: nvd_CVE-2023-4966].\n\n"
        "**Next Checks**\n"
        "- Audit active NetScaler sessions [Source: cisa_kev_CVE-2023-4966].\n"
    )
    chain, fake_llm = _make_extended(llm_text, [_nvd_rchunk(), _kev_rchunk()])

    resp = chain.query("What is the impact of CVE-2023-4966 on Citrix NetScaler?")

    system_msg = fake_llm.last_messages[0].content
    # Unchanged template prompt present, assist layer appended at the end.
    assert "Triage Card mode" in system_msg
    assert system_msg.endswith(_ANALYST_ASSIST_INSTRUCTIONS)
    assert "Detection & Hunting" in system_msg
    # L1 card + normalized L2 output composed as usual.
    assert resp.l1_block.startswith("## CVE-2023-4966")
    assert "[Source: nvd_CVE-2023-4966]" in resp.l2_output
    assert resp.answer.startswith(resp.l1_block)


def test_extended_chain_deepens_retrieval_via_instance_override():
    chain, _ = _make_extended("text", [_nvd_rchunk(), _kev_rchunk()])
    assert chain.retriever.rerank_top_k == _EXTENDED_RERANK_TOP_K
    assert _EXTENDED_RERANK_TOP_K > 5


def test_extended_chain_abstains_without_hosted_call():
    chain, fake_llm = _make_extended("should not be invoked", [_nvd_rchunk()])

    resp = chain.query("Summarize CVE-2099-12345.")

    assert resp.abstention_reason is not None
    assert resp.generation_time_ms == 0.0
    assert fake_llm.last_messages is None  # hosted model never invoked
