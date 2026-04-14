"""
Unit tests for entity-group aggregation and fact-bundle construction.

These tests use synthetic chunk dictionaries shaped like the output of
:func:`cti_rag.rag.chain._build_chunk_dict`. The goal is to exercise the
deterministic Layer-1 logic end-to-end without needing a real retriever.
"""

import json

from src.cti_rag.rag.facts import (
    FactBundle,
    build_fact_bundle,
    build_threat_context_facts,
    build_vuln_triage_facts,
    extract_cve_ids,
    group_chunks_by_cve,
    group_chunks_by_technique,
)
from src.cti_rag.rag.severity import annotate_vuln_triage_bundle


def _nvd_chunk(cve_id="CVE-2023-4966", **overrides):
    chunk = {
        "doc_id": f"nvd_{cve_id}",
        "title": f"NVD: {cve_id}",
        "content": f"{cve_id} is a buffer overflow in Citrix NetScaler.",
        "source": "nvd",
        "citation_label": f"NVD: {cve_id} | nvd_{cve_id}",
        "metadata": {
            "source": "nvd",
            "title": f"NVD: {cve_id}",
            "cvss_score": 9.4,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
            "severity": "critical",
            "cwe_ids": "CWE-119",
            "cve_ids": cve_id,
            "affected_products": (
                "cpe:2.3:a:citrix:netscaler_adc:*:*:*:*:*:*:*:*,"
                "cpe:2.3:a:citrix:netscaler_gateway:*:*:*:*:*:*:*:*"
            ),
            "references": "https://example.org/advisory",
        },
    }
    chunk["metadata"].update(overrides.get("metadata", {}))
    chunk.update({k: v for k, v in overrides.items() if k != "metadata"})
    return chunk


def _kev_chunk(cve_id="CVE-2023-4966", ransomware=True):
    return {
        "doc_id": f"cisa_kev_{cve_id}",
        "title": f"CISA KEV: {cve_id}",
        "content": f"{cve_id} is actively exploited.",
        "source": "cisa_kev",
        "citation_label": f"CISA KEV: {cve_id} | cisa_kev_{cve_id}",
        "metadata": {
            "source": "cisa_kev",
            "title": f"CISA KEV: {cve_id}",
            "cve_ids": cve_id,
            "date_added_to_kev": "2023-10-18",
            "due_date": "2023-11-08",
            "required_action": "Apply mitigations per vendor instructions.",
            "known_ransomware": "Known" if ransomware else "Unknown",
            "known_ransomware_bool": ransomware,
        },
    }


def _misp_chunk(misp_id="abc-123", cve_ids="CVE-2024-24919", technique="T1190"):
    iocs = [
        {"type": "ip-dst", "value": "198.51.100.10", "category": "Network activity"},
        {"type": "domain", "value": "malicious.example.org", "category": "Network activity"},
        {"type": "sha256", "value": "a" * 64, "category": "Payload delivery"},
    ]
    return {
        "doc_id": f"misp_{misp_id}",
        "title": f"MISP Event {misp_id}",
        "content": f"Exploitation of {cve_ids} via {technique}.",
        "source": "misp",
        "citation_label": f"MISP Event {misp_id} | misp_{misp_id}",
        "metadata": {
            "source": "misp",
            "title": f"MISP Event {misp_id}",
            "cve_ids": cve_ids,
            "attack_techniques": technique,
            "iocs_json": json.dumps(iocs),
            "iocs_count": len(iocs),
            "misp_uuid": misp_id,
        },
    }


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_extract_cve_ids_normalizes_case_and_deduplicates():
    text = "cve-2024-1234 is related to CVE-2024-1234 and CVE-2023-4966."
    assert extract_cve_ids(text) == ["CVE-2024-1234", "CVE-2024-1234", "CVE-2023-4966"]


# ---------------------------------------------------------------------------
# Entity-group aggregation
# ---------------------------------------------------------------------------


def test_group_chunks_by_cve_separates_primary_and_supporting():
    primary = _nvd_chunk("CVE-2023-4966")
    supporting = _misp_chunk(cve_ids="CVE-2023-4966")  # mentions CVE but title is MISP

    groups, supporting_chunks = group_chunks_by_cve(
        [primary, supporting], ["CVE-2023-4966"]
    )

    assert groups["CVE-2023-4966"] == [primary]
    assert supporting_chunks == [supporting]


def test_group_chunks_by_cve_allows_multi_group_membership():
    # An advisory that is titled for two CVEs should be primary for both.
    dual_title_chunk = {
        "doc_id": "advisory_AA26-002A",
        "title": "Advisory on CVE-2023-46805 and CVE-2024-21887",
        "content": "Chained exploit in Ivanti products.",
        "source": "cisa_advisory",
        "citation_label": "Advisory AA26-002A | advisory_AA26-002A",
        "metadata": {"source": "cisa_advisory"},
    }

    groups, _ = group_chunks_by_cve(
        [dual_title_chunk], ["CVE-2023-46805", "CVE-2024-21887"]
    )

    assert groups["CVE-2023-46805"] == [dual_title_chunk]
    assert groups["CVE-2024-21887"] == [dual_title_chunk]


def test_group_chunks_by_technique_splits_primary_and_supporting():
    with_technique = _misp_chunk(technique="T1190")
    without_technique = _misp_chunk(misp_id="xyz-999", technique="T1059")

    primary, supporting = group_chunks_by_technique(
        [with_technique, without_technique], "T1190"
    )
    assert primary == [with_technique]
    assert supporting == [without_technique]


# ---------------------------------------------------------------------------
# VulnTriage facts
# ---------------------------------------------------------------------------


def test_build_vuln_triage_facts_merges_nvd_and_kev_signals():
    facts = build_vuln_triage_facts(
        "CVE-2023-4966", [_nvd_chunk(), _kev_chunk()]
    )

    assert facts.cve_id == "CVE-2023-4966"
    assert facts.cvss_score == 9.4
    assert facts.severity == "critical"
    assert "CWE-119" in facts.cwe_ids
    assert facts.kev_listed is True
    assert facts.kev_date_added == "2023-10-18"
    assert facts.kev_due_date == "2023-11-08"
    assert facts.ransomware_use is True
    assert "Citrix NetScaler ADC" in facts.affected_products
    assert "Citrix NetScaler Gateway" in facts.affected_products
    assert "nvd" in facts.sources and "cisa_kev" in facts.sources
    assert len(facts.source_chunks) == 2


def test_build_vuln_triage_facts_flags_cvss_inconsistency_across_sources():
    alt = _nvd_chunk()
    alt["metadata"]["cvss_score"] = 7.5  # pretend a second NVD-style chunk disagrees
    alt["doc_id"] = "nvd_alt"
    alt["title"] = "NVD alt: CVE-2023-4966"

    facts = build_vuln_triage_facts("CVE-2023-4966", [_nvd_chunk(), alt])

    cvss_issues = [i for i in facts.inconsistencies if i["field"] == "cvss_score"]
    assert cvss_issues, "expected CVSS inconsistency to be flagged"


def test_build_vuln_triage_facts_handles_cve_not_in_kev():
    facts = build_vuln_triage_facts("CVE-2024-99999", [_nvd_chunk("CVE-2024-99999")])

    assert facts.kev_listed is False
    assert facts.ransomware_use is False
    assert facts.kev_date_added is None


# ---------------------------------------------------------------------------
# ThreatContext facts
# ---------------------------------------------------------------------------


def test_build_threat_context_facts_aggregates_iocs_and_cves():
    chunk_a = _misp_chunk(misp_id="event-1", cve_ids="CVE-2024-24919", technique="T1190")
    chunk_b = _misp_chunk(misp_id="event-2", cve_ids="CVE-2024-00001", technique="T1190")
    # Override IoCs in chunk_b to contain a duplicate + one new domain.
    chunk_b["metadata"]["iocs_json"] = json.dumps(
        [
            {"type": "ip-dst", "value": "198.51.100.10", "category": None},
            {"type": "domain", "value": "another.example.org", "category": None},
        ]
    )

    facts = build_threat_context_facts("T1190", "technique", [chunk_a, chunk_b])

    assert facts.attack_technique == "T1190"
    assert "CVE-2024-24919" in facts.associated_cves
    assert "CVE-2024-00001" in facts.associated_cves
    # De-duplicated IP.
    assert facts.iocs["ip"] == ["198.51.100.10"]
    assert len(facts.iocs["domain"]) == 2
    assert facts.iocs_count == 4  # 1 ip + 2 domains + 1 hash


def test_build_threat_context_facts_resolves_tactic_name():
    facts = build_threat_context_facts("TA0001", "tactic", [])
    assert facts.tactic_id == "TA0001"
    assert facts.tactic_name == "Initial Access"


# ---------------------------------------------------------------------------
# Top-level dispatch + severity annotation
# ---------------------------------------------------------------------------


def test_build_fact_bundle_vuln_triage_populates_vuln_list():
    chunks = [_nvd_chunk(), _kev_chunk()]
    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks)

    assert isinstance(bundle, FactBundle)
    assert bundle.template == "VulnTriage"
    assert len(bundle.vuln_triage) == 1
    assert bundle.threat_context is None
    assert bundle.cross_source is None


def test_build_fact_bundle_vuln_triage_multi_cve_renders_two_bundles():
    chunks = [_nvd_chunk("CVE-2023-46805"), _nvd_chunk("CVE-2024-21887")]
    bundle = build_fact_bundle(
        "VulnTriage", ["CVE-2023-46805", "CVE-2024-21887"], chunks
    )

    ids = {f.cve_id for f in bundle.vuln_triage}
    assert ids == {"CVE-2023-46805", "CVE-2024-21887"}


def test_build_fact_bundle_cross_source_preserves_source_coverage():
    chunks = [_nvd_chunk("CVE-2023-4966"), _kev_chunk("CVE-2023-4966")]
    bundle = build_fact_bundle(
        "CrossSourceCompare", ["CVE-2023-4966", "CVE-2024-99999"], chunks
    )

    assert bundle.cross_source is not None
    coverage = bundle.cross_source.source_coverage
    assert set(coverage["CVE-2023-4966"]) == {"nvd", "cisa_kev"}
    # CVE-2024-99999 was requested but no chunks matched — still listed.
    assert coverage["CVE-2024-99999"] == []


def test_annotate_vuln_triage_bundle_attaches_critical_signal():
    facts = build_vuln_triage_facts("CVE-2023-4966", [_nvd_chunk(), _kev_chunk()])
    annotate_vuln_triage_bundle(facts)

    assert facts.triage_signal is not None
    assert facts.triage_signal["severity"] == "critical"
    assert "KEV" in facts.triage_signal["rationale"]


def test_annotate_vuln_triage_bundle_marks_unknown_when_cvss_missing():
    chunk = _kev_chunk("CVE-2024-00001", ransomware=False)  # no CVSS in KEV
    facts = build_vuln_triage_facts("CVE-2024-00001", [chunk])
    annotate_vuln_triage_bundle(facts)

    assert facts.triage_signal["severity"] == "unknown"


def test_build_fact_bundle_threat_context_uses_technique_filter():
    relevant = _misp_chunk(technique="T1190")
    noise = _misp_chunk(misp_id="noise", technique="T1059")

    bundle = build_fact_bundle(
        "ThreatContext",
        ["T1190"],
        [relevant, noise],
        threat_context_kind="technique",
    )

    tc = bundle.threat_context
    assert tc is not None
    assert tc.attack_technique == "T1190"
    # Only the relevant chunk feeds iocs / cves; the noise chunk is
    # demoted to supporting.
    assert "CVE-2024-24919" in tc.associated_cves
    assert bundle.supporting_chunks == [noise]
