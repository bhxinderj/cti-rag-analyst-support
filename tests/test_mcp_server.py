"""
Unit tests for the MCP facts pipeline (src/cti_rag/mcp/server.py).

The pipeline reuses the evaluated Phase-2 building blocks import-only
and never invokes an LLM — so it is fully testable with the fake
retriever fixtures from the templated-chain tests.
"""

from __future__ import annotations

from src.cti_rag.mcp.server import run_facts_pipeline
from tests.test_chain_templated import (
    _FakeCollection,
    _FakeRetriever,
    _kev_rchunk,
    _misp_rchunk,
    _nvd_rchunk,
)


def test_facts_pipeline_renders_vuln_triage_card_without_llm():
    retriever = _FakeRetriever(
        [_nvd_rchunk(), _kev_rchunk()], collection=_FakeCollection({})
    )

    out = run_facts_pipeline(
        "What is the impact of CVE-2023-4966 on Citrix NetScaler?", retriever
    )

    assert "CVE-2023-4966" in out
    assert "Triage Signal" in out
    assert "Sources (2)" in out
    assert "nvd_CVE-2023-4966" in out
    assert "no model invoked" in out


def test_facts_pipeline_abstains_on_unknown_entity():
    other = _misp_rchunk()
    other.metadata["cve_ids"] = "CVE-2020-9999"
    other.content = "Unrelated MISP event."
    retriever = _FakeRetriever([other], collection=_FakeCollection({}))

    out = run_facts_pipeline("Summarize CVE-2099-12345.", retriever)

    assert "No sufficiently grounded evidence" in out
    assert "cve-2099-12345" in out.lower()
    assert "no model invoked" in out
