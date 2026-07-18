"""
Unit tests for ``RAGChain.query_templated`` — the Phase-2 wire-in.

Strategy
--------
We do NOT invoke Ollama, and we do NOT hit Chroma. Each test constructs
a real :class:`RAGChain` with ``ChatOllama`` temporarily replaced by a
no-op dummy, then attaches a hand-written ``_FakeLLM`` and
``_FakeRetriever`` to the instance. The assertions focus on:

- routing → template assignment
- L1 markdown assembly (passed through verbatim into ``response.answer``)
- LLM prompt shape (the target-section block the model must follow)
- citation normalization on the generated L2 output
- Phase-2 fields added to :class:`RAGResponse`
- entity-aware recall augmentation fallback

A separate integration test (``tests/run_live_e2e_smoke.py``) covers the
real Ollama + Chroma path.

Tests follow this repo's convention: plain ``def test_*`` functions with
bare ``assert`` statements, no pytest fixtures. They are discovered by
the same importlib reflection loop used for the other ``test_*.py``
modules.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterable

from src.cti_rag.rag import chain as chain_module
from src.cti_rag.rag.chain import RAGChain
from src.cti_rag.retrieval.hybrid_retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Captures the last invocation and returns a fixed markdown block."""

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_messages: list | None = None

    def invoke(self, messages):
        self.last_messages = list(messages)
        return SimpleNamespace(content=self._response_text)


class _FakeCollection:
    """Stub for ``retriever.collection`` used by entity-aware augmentation.

    ``_augment_chunks_for_cves`` calls ``collection.get(where={"cve_ids":
    {"$eq": CVE}}, limit=20)``. We return an empty payload by default so
    the augmentation path is a no-op; individual tests can override.
    """

    def __init__(self, extra_chunks: dict[str, list[dict]] | None = None):
        self._extra = extra_chunks or {}

    def get(self, where, limit=20):  # noqa: ARG002
        cve = (where or {}).get("cve_ids", {}).get("$eq", "")
        hits = self._extra.get(cve.upper(), [])
        return {
            "ids": [h["doc_id"] for h in hits],
            "documents": [h["content"] for h in hits],
            "metadatas": [h["metadata"] for h in hits],
        }


class _FakeRetriever:
    """Returns a canned list of ``RetrievedChunk`` objects."""

    def __init__(self, chunks: Iterable[RetrievedChunk], collection=None):
        self._chunks = list(chunks)
        self.last_trace: dict = {"fake": True}
        self.collection = collection or _FakeCollection()

    def retrieve(self, _query: str) -> list[RetrievedChunk]:
        return list(self._chunks)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@contextmanager
def _patched_chat_ollama():
    """Swap ``ChatOllama`` for a dummy so ``RAGChain.__init__`` is safe."""

    class _DummyChatOllama:
        def __init__(self, *_args, **_kwargs):
            pass

        def invoke(self, _messages):  # pragma: no cover — replaced on instance
            raise AssertionError("_DummyChatOllama.invoke should not be called")

    original = chain_module.ChatOllama
    chain_module.ChatOllama = _DummyChatOllama
    try:
        yield
    finally:
        chain_module.ChatOllama = original


def _nvd_rchunk(cve_id: str = "CVE-2023-4966", score: float = 0.91, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=f"nvd_{cve_id}",
        content=f"{cve_id} is an information-disclosure vulnerability in Citrix NetScaler.",
        metadata={
            "source": "nvd",
            "title": f"NVD: {cve_id}",
            "cvss_score": 7.5,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
            "severity": "high",
            "cwe_ids": "CWE-119",
            "cve_ids": cve_id,
            "affected_products": "cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*",
        },
        score=score,
        rank=rank,
    )


def _kev_rchunk(cve_id: str = "CVE-2023-4966", score: float = 0.88, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=f"cisa_kev_{cve_id}",
        content=f"{cve_id} is actively exploited per CISA KEV.",
        metadata={
            "source": "cisa_kev",
            "title": f"CISA KEV: {cve_id}",
            "cve_ids": cve_id,
            "date_added_to_kev": "2023-10-18",
            "due_date": "2023-11-08",
            "required_action": "Apply vendor patches per Citrix advisory.",
            "known_ransomware": "Known",
            "known_ransomware_bool": True,
        },
        score=score,
        rank=rank,
    )


def _misp_rchunk(tech: str = "T1190", score: float = 0.80, rank: int = 2) -> RetrievedChunk:
    iocs = [
        {"type": "ip-dst", "value": "198.51.100.10", "category": None},
        {"type": "domain", "value": "malicious.example.org", "category": None},
    ]
    return RetrievedChunk(
        doc_id="misp_abc123",
        content=f"Exploitation of CVE-2023-4966 via {tech}.",
        metadata={
            "source": "misp",
            "title": "MISP Event abc123",
            "cve_ids": "CVE-2023-4966",
            "attack_techniques": tech,
            "iocs_json": json.dumps(iocs),
            "iocs_count": len(iocs),
        },
        score=score,
        rank=rank,
    )


def _make_chain(
    *,
    llm_response: str,
    chunks: list[RetrievedChunk],
    extra_by_cve: dict[str, list[dict]] | None = None,
) -> tuple[RAGChain, _FakeLLM, _FakeRetriever]:
    with _patched_chat_ollama():
        chain = RAGChain(retrieval_mode="hybrid")

    fake_llm = _FakeLLM(llm_response)
    fake_retriever = _FakeRetriever(
        chunks, collection=_FakeCollection(extra_by_cve or {})
    )
    chain.llm = fake_llm
    chain.retriever = fake_retriever
    return chain, fake_llm, fake_retriever


# ---------------------------------------------------------------------------
# Tests — VulnTriage path
# ---------------------------------------------------------------------------


def test_query_templated_vuln_triage_routes_and_assembles():
    llm_text = (
        "**Exploitation Context**\n"
        "- Memory overread exposes session tokens [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n\n"
        "**Affected Versions**\n"
        "- Citrix NetScaler ADC versions before 13.1-49.15 [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n\n"
        "**Mitigations**\n"
        "- Apply vendor patch per Citrix advisory [Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n\n"
        "**Evidence**\n"
        "- NVD: CVE-2023-4966 — vulnerability record [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n\n"
        "**Gaps**\n"
        "- No IoCs or detection rules in retrieved context.\n"
    )
    chain, fake_llm, _ = _make_chain(
        llm_response=llm_text,
        chunks=[_nvd_rchunk(), _kev_rchunk()],
    )

    resp = chain.query_templated("What is the impact of CVE-2023-4966 on Citrix NetScaler?")

    # Routing
    assert resp.template == "VulnTriage"
    assert resp.routing_decision["template"] == "VulnTriage"
    assert resp.routing_decision["primary_entities"] == ["CVE-2023-4966"]
    assert resp.routing_decision["rule_matched"] == 3

    # L1 block is deterministic markdown with the triage signal — and
    # it's prepended verbatim to the final answer.
    assert resp.l1_block.startswith("## CVE-2023-4966")
    assert "CVSS: 7.5" in resp.l1_block
    assert "Listed in CISA KEV: yes" in resp.l1_block
    assert "Known ransomware use: yes" in resp.l1_block
    assert resp.answer.startswith(resp.l1_block)

    # L2 output: the LLM's text, with citation normalization applied.
    # Canonical labels are bare doc_ids so small LLMs can reproduce them;
    # the long-form citation in the fixture resolves via doc_id substring.
    assert "**Exploitation Context**" in resp.l2_output
    assert "[Source: nvd_CVE-2023-4966]" in resp.l2_output

    # fact_bundle is serialized and carries the VulnTriage entity.
    assert resp.fact_bundle["template"] == "VulnTriage"
    assert resp.fact_bundle["vuln_triage"][0]["cve_id"] == "CVE-2023-4966"
    assert resp.fact_bundle["vuln_triage"][0]["triage_signal"] is not None

    # Prompt shape: the LLM was asked for the five target sections.
    system_msg, user_msg = fake_llm.last_messages
    assert "Triage Card mode" in system_msg.content
    assert "**Exploitation Context** (L2)" in user_msg.content
    assert "**Evidence** (L1)" in user_msg.content
    # Allowed citation labels are listed.
    assert "nvd_CVE-2023-4966" in user_msg.content

    # Source documents are annotated with cited_in_answer.
    nvd_doc = next(d for d in resp.source_documents if d["doc_id"] == "nvd_CVE-2023-4966")
    assert nvd_doc["cited_in_answer"] is True


def test_query_templated_drops_invalid_citations():
    llm_text = (
        "**Exploitation Context**\n"
        "- Claim with a fabricated source [Source: Totally Made Up].\n"
        "- Claim backed by a real label [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n"
    )
    chain, _, _ = _make_chain(
        llm_response=llm_text,
        chunks=[_nvd_rchunk(), _kev_rchunk()],
    )

    resp = chain.query_templated("Tell me about CVE-2023-4966.")

    # The fabricated citation is stripped; the real one survives (normalized
    # to the short canonical doc_id label).
    assert "Totally Made Up" not in resp.l2_output
    assert "[Source: nvd_CVE-2023-4966]" in resp.l2_output


# ---------------------------------------------------------------------------
# Tests — ThreatContext path
# ---------------------------------------------------------------------------


def test_query_templated_threat_context_technique():
    llm_text = (
        "**Threat Actors and Malware**\n"
        "- No attribution in retrieved context.\n\n"
        "**Defensive Guidance**\n"
        "- Monitor for unusual outbound requests [Source: MISP Event abc123 | misp_abc123].\n\n"
        "**Evidence**\n"
        "- MISP Event abc123 — exploitation pattern [Source: MISP Event abc123 | misp_abc123].\n\n"
        "**Gaps**\n"
        "- No YARA or Sigma rules present.\n"
    )
    chain, fake_llm, _ = _make_chain(
        llm_response=llm_text,
        chunks=[_misp_rchunk(tech="T1190")],
    )

    resp = chain.query_templated("What ATT&CK coverage exists for T1190?")

    assert resp.template == "ThreatContext"
    assert resp.routing_decision["threat_context_kind"] == "technique"
    assert resp.routing_decision["primary_entities"] == ["T1190"]

    # L1 renders the technique header and IoC summary.
    assert "T1190" in resp.l1_block
    assert "IoC Summary" in resp.l1_block

    # Prompt asks for the ThreatContext sections.
    user_msg = fake_llm.last_messages[1].content
    assert "**Threat Actors and Malware** (L2)" in user_msg
    assert "**Defensive Guidance** (L2)" in user_msg


# ---------------------------------------------------------------------------
# Tests — CrossSourceCompare entity fallback
# ---------------------------------------------------------------------------


def test_query_templated_cross_source_auto_detects_cves():
    llm_text = (
        "**Source Agreement**\n"
        "- NVD and CISA KEV agree that CVE-2023-4966 is critical "
        "[Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966] "
        "[Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n\n"
        "**Cross-Source Facts**\n"
        "- Only KEV records active exploitation "
        "[Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n\n"
        "**Evidence**\n"
        "- NVD entry [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n\n"
        "**Gaps**\n"
        "- No MISP attribution retrieved.\n"
    )
    # Query names both NVD and CISA → router rule 1 fires with no CVEs.
    chain, _, _ = _make_chain(
        llm_response=llm_text,
        chunks=[_nvd_rchunk(), _kev_rchunk()],
    )

    resp = chain.query_templated(
        "Compare what NVD and CISA KEV say about recent Citrix issues."
    )

    assert resp.template == "CrossSourceCompare"
    # No explicit CVEs in the query, so primary_entities should be derived
    # from the retrieved chunk mentions.
    assert "CVE-2023-4966" in resp.fact_bundle["primary_entities"]
    # The comparison table L1 block names the CVE.
    assert "CVE-2023-4966" in resp.l1_block
    assert "Comparison Table" in resp.l1_block


# ---------------------------------------------------------------------------
# Tests — abstention on thin context
# ---------------------------------------------------------------------------


def test_query_templated_abstains_when_entity_missing():
    """If the query names a CVE absent from retrieved context, we abstain."""
    other = _misp_rchunk()
    other.metadata["cve_ids"] = "CVE-2020-9999"
    other.content = "Unrelated MISP event."
    chain, _, _ = _make_chain(
        llm_response="should not be invoked",
        chunks=[other],
    )

    resp = chain.query_templated("Summarize CVE-2099-12345.")

    assert resp.abstention_reason is not None
    assert "cve-2099-12345" in resp.abstention_reason.lower()
    # Legacy Phase-1 abstention format is preserved so existing consumers
    # keep working.
    assert resp.answer.startswith("Summary:")
    # Still carries routing context even in the abstention path.
    assert resp.template == "VulnTriage"


# ---------------------------------------------------------------------------
# Tests — entity-aware augmentation fallback
# ---------------------------------------------------------------------------


def test_query_templated_entity_aware_augmentation_fills_missing_nvd():
    """When the ranker misses the primary NVD chunk, augmentation fills it."""
    # Retrieval returns only MISP + KEV — no NVD chunk.
    misp = _misp_rchunk()
    kev = _kev_rchunk()
    nvd_dict = {
        "doc_id": "nvd_CVE-2023-4966",
        "content": "NVD record for CVE-2023-4966.",
        "metadata": {
            "source": "nvd",
            "title": "NVD: CVE-2023-4966",
            "cvss_score": 7.5,
            "cwe_ids": "CWE-119",
            "cve_ids": "CVE-2023-4966",
        },
    }
    llm_text = (
        "**Exploitation Context**\n"
        "- Buffer overread [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n"
    )

    chain, _, _ = _make_chain(
        llm_response=llm_text,
        chunks=[misp, kev],
        extra_by_cve={"CVE-2023-4966": [nvd_dict]},
    )

    resp = chain.query_templated("Brief me on CVE-2023-4966.")

    # NVD chunk was pulled in by the entity-aware fallback.
    doc_ids = {d["doc_id"] for d in resp.source_documents}
    assert "nvd_CVE-2023-4966" in doc_ids
    # Trace records the augmentation.
    assert resp.retrieval_trace.get("entity_aware_augmented_cves") == ["CVE-2023-4966"]


def test_query_templated_augmentation_respects_total_chunk_cap():
    """Augmentation must never grow the context past the hard ceiling."""
    from src.cti_rag.rag.chain import _AUGMENT_MAX_TOTAL_CHUNKS, _augment_chunks_for_cves

    base_chunks = [
        {
            "doc_id": f"misp_base_{idx}",
            "content": "Base chunk.",
            "source": "misp",
            "title": f"Base {idx}",
            "metadata": {"source": "misp"},
        }
        for idx in range(5)
    ]
    cves = [f"CVE-2024-{1000 + idx}" for idx in range(8)]
    extra = {
        cve: [
            {
                "doc_id": f"nvd_{cve}",
                "content": f"NVD record for {cve}.",
                "metadata": {"source": "nvd", "title": f"NVD: {cve}", "cve_ids": cve},
            },
            {
                "doc_id": f"cisa_kev_{cve}",
                "content": f"KEV record for {cve}.",
                "metadata": {"source": "cisa_kev", "title": f"KEV: {cve}", "cve_ids": cve},
            },
        ]
        for cve in cves
    }
    retriever = _FakeRetriever(chunks=[], collection=_FakeCollection(extra))

    augmented, augmented_cves = _augment_chunks_for_cves(retriever, base_chunks, cves)

    assert len(augmented) <= _AUGMENT_MAX_TOTAL_CHUNKS
    # The cap still leaves room for the first CVEs to be augmented.
    assert augmented_cves
