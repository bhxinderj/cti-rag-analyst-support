"""
Unit tests for the Triage Field Coverage evaluator.

Strategy
--------
Every test builds a synthetic fact-bundle dict (matching the shape
produced by ``dataclasses.asdict(FactBundle(...))``) and a synthetic
rendered-answer string, then hands them to ``evaluate_field_coverage``.
No LLM, no retrieval, no Chroma.

Each calibrated ``required_fields`` schema (vuln_004, ttp_001, cross_003)
gets direct coverage, plus edge cases around the L2 bullet counter
(empty-section fallbacks, uncited bullets, continuation lines).

Tests follow this repo's convention: plain ``def test_*`` functions with
bare ``assert`` statements, no pytest fixtures. They are discovered by
the same importlib reflection loop used for the other ``test_*.py``
modules.
"""

from __future__ import annotations

from src.cti_rag.evaluation.field_coverage import (
    FieldResult,
    aggregate_field_coverage,
    evaluate_field_coverage,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic bundles shaped like ``asdict(FactBundle)`` output.
# ---------------------------------------------------------------------------


def _vuln_bundle(
    *,
    cve_id: str = "CVE-2023-4966",
    cvss: float | None = 7.5,
    cwe_ids: tuple[str, ...] = ("CWE-119",),
    affected: tuple[str, ...] = (
        "Citrix NetScaler ADC",
        "Citrix NetScaler Gateway",
    ),
    kev: bool = True,
    ransomware: bool = True,
    triage: str = "high",
) -> dict:
    return {
        "template": "VulnTriage",
        "primary_entities": [cve_id],
        "vuln_triage": [
            {
                "cve_id": cve_id,
                "cvss_score": cvss,
                "cwe_ids": list(cwe_ids),
                "affected_products": list(affected),
                "kev_listed": kev,
                "ransomware_use": ransomware,
                "triage_signal": {"severity": triage, "label": "HIGH — Prioritize"},
                "sources": ["nvd", "cisa_kev"],
                "source_chunks": [],
                "inconsistencies": [],
            }
        ],
        "threat_context": None,
        "cross_source": None,
        "supporting_chunks": [],
    }


def _threat_bundle(
    *,
    technique: str = "T1190",
    tactic_id: str = "TA0001",
    tactic_name: str = "Initial Access",
    associated_cves: tuple[str, ...] = ("CVE-2023-4966", "CVE-2023-22515"),
) -> dict:
    return {
        "template": "ThreatContext",
        "primary_entities": [technique],
        "vuln_triage": [],
        "threat_context": {
            "primary_entity": technique,
            "primary_entity_kind": "technique",
            "attack_technique": technique,
            "tactic_id": tactic_id,
            "tactic_name": tactic_name,
            "associated_cves": list(associated_cves),
            "iocs": {},
            "iocs_count": 0,
            "sources": ["misp"],
            "source_chunks": [],
        },
        "cross_source": None,
        "supporting_chunks": [],
    }


def _cross_bundle(
    *,
    entities: tuple[str, ...] = (
        "CVE-2023-22515",
        "CVE-2023-34362",
        "CVE-2023-3519",
    ),
) -> dict:
    return {
        "template": "CrossSourceCompare",
        "primary_entities": list(entities),
        "vuln_triage": [],
        "threat_context": None,
        "cross_source": {
            "entities": [
                {
                    "cve_id": cve,
                    "cvss_score": 9.8,
                    "sources": ["nvd", "cisa_kev"],
                }
                for cve in entities
            ],
            "source_coverage": {cve: ["nvd", "cisa_kev"] for cve in entities},
            "comparison_axes": [],
        },
        "supporting_chunks": [],
    }


# ---------------------------------------------------------------------------
# vuln_004 schema (VulnTriage) — L1 + L2 min_count
# ---------------------------------------------------------------------------


_VULN_004_REQ = {
    "cve_id": "CVE-2023-4966",
    "cvss_score": 7.5,
    "cwe_ids": ["CWE-119"],
    "affected_products": ["Citrix NetScaler ADC", "Citrix NetScaler Gateway"],
    "kev_listed": True,
    "ransomware_use": True,
    "triage_signal": "high",
    "mitigations_min_count": 1,
}


def test_vuln_004_all_pass_on_clean_bundle_and_answer():
    answer = (
        "## CVE-2023-4966\n\n"
        "**Triage Signal:** HIGH — Prioritize\n\n"
        "**Mitigations** (L2)\n"
        "- Apply Citrix vendor patch [Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n"
        "- Kill active sessions [Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n"
    )
    result = evaluate_field_coverage(
        "vuln_004",
        _VULN_004_REQ,
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=answer,
    )
    assert result.fail_count == 0, [r for r in result.results if r.status == "FAIL"]
    assert result.pass_count == 8
    assert result.coverage_ratio == 1.0


def test_vuln_004_fails_on_wrong_cvss_and_empty_mitigations():
    answer = (
        "**Mitigations** (L2)\n"
        "- No reliable mitigation guidance is present in the retrieved context.\n"
    )
    result = evaluate_field_coverage(
        "vuln_004",
        _VULN_004_REQ,
        template="VulnTriage",
        fact_bundle=_vuln_bundle(cvss=5.5, triage="moderate"),
        answer=answer,
    )
    failed = {r.field_name for r in result.results if r.status == "FAIL"}
    assert "cvss_score" in failed
    assert "triage_signal" in failed
    assert "mitigations_min_count" in failed


def test_vuln_004_uncited_bullets_do_not_count_for_l2_min_count():
    # The bullet exists, but without a [Source: ...] citation → uncounted.
    answer = (
        "**Mitigations** (L2)\n"
        "- Install the patch immediately.\n"
        "- Rotate all session cookies.\n"
    )
    result = evaluate_field_coverage(
        "vuln_004",
        {"mitigations_min_count": 1},
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=answer,
    )
    [r] = result.results
    assert r.field_name == "mitigations_min_count"
    assert r.status == "FAIL"
    assert r.actual == 0


def test_vuln_004_continuation_line_citation_counts_for_its_bullet():
    answer = (
        "**Mitigations** (L2)\n"
        "- Apply mitigations and kill all active and persistent sessions\n"
        "  per vendor instructions\n"
        "  [Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n"
    )
    result = evaluate_field_coverage(
        "vuln_004",
        {"mitigations_min_count": 1},
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=answer,
    )
    [r] = result.results
    assert r.status == "PASS", r.detail
    assert r.actual == 1


# ---------------------------------------------------------------------------
# ttp_001 schema (ThreatContext) — attack aggregation + L2 detection
# ---------------------------------------------------------------------------


_TTP_001_REQ = {
    "attack_techniques": ["T1190"],
    "attack_tactic": ["TA0001", "Initial Access"],
    "associated_cves_min_count": 1,
    "detection_or_mitigation_min_count": 1,
}


def test_ttp_001_all_pass_when_bundle_and_defensive_section_are_populated():
    answer = (
        "**Defensive Guidance** (L2)\n"
        "- Monitor unusual outbound requests from public-facing apps "
        "[Source: MISP Event abc | misp_abc].\n"
        "- Patch exposed VPN appliances promptly "
        "[Source: MISP Event def | misp_def].\n"
    )
    result = evaluate_field_coverage(
        "ttp_001",
        _TTP_001_REQ,
        template="ThreatContext",
        fact_bundle=_threat_bundle(),
        answer=answer,
    )
    assert result.fail_count == 0, [r for r in result.results if r.status == "FAIL"]
    assert result.pass_count == 4


def test_ttp_001_falls_back_to_mitigations_section_when_defensive_absent():
    # Some prompts may land on "Mitigations" instead of "Defensive Guidance".
    # The mapping should accept either.
    answer = (
        "**Mitigations** (L2)\n"
        "- Deploy WAF rules for CVE-2023-4966 [Source: MISP Event abc | misp_abc].\n"
    )
    result = evaluate_field_coverage(
        "ttp_001",
        {"detection_or_mitigation_min_count": 1},
        template="ThreatContext",
        fact_bundle=_threat_bundle(),
        answer=answer,
    )
    [r] = result.results
    assert r.status == "PASS", r.detail
    assert r.actual == 1


def test_ttp_001_attack_techniques_aggregates_from_source_documents():
    # The bundle alone only carries the primary technique; the evaluator
    # should expand via source_documents metadata.
    bundle = _threat_bundle()
    bundle["threat_context"]["attack_technique"] = "T1190"
    source_docs = [
        {
            "title": "MISP",
            "content": "Phishing with T1566.001",
            "metadata": {"attack_techniques": "T1566,T1566.001"},
        }
    ]
    result = evaluate_field_coverage(
        "ttp_001",
        {"attack_techniques": ["T1566.001"]},
        template="ThreatContext",
        fact_bundle=bundle,
        answer="",
        source_documents=source_docs,
    )
    [r] = result.results
    assert r.status == "PASS", r.detail
    assert "T1566.001" in r.actual


# ---------------------------------------------------------------------------
# cross_003 schema (CrossSourceCompare) — entity-match + L2 facts
# ---------------------------------------------------------------------------


_CROSS_003_REQ = {
    "expected_entities": ["CVE-2023-22515", "CVE-2023-34362", "CVE-2023-3519"],
    "expected_entities_min_matches": 2,
    "cross_source_facts_min_count": 2,
}


def test_cross_003_all_pass_with_matching_bundle_and_l2_bullets():
    answer = (
        "**Cross-Source Facts** (L2)\n"
        "- NVD records CVSS 9.8 [Source: NVD: CVE-2023-22515 | nvd_CVE-2023-22515] "
        "while CISA KEV confirms exploitation "
        "[Source: CISA KEV: CVE-2023-22515 | cisa_kev_CVE-2023-22515].\n"
        "- Only KEV flags ransomware use [Source: CISA KEV: CVE-2023-34362 | cisa_kev_CVE-2023-34362].\n"
    )
    result = evaluate_field_coverage(
        "cross_003",
        _CROSS_003_REQ,
        template="CrossSourceCompare",
        fact_bundle=_cross_bundle(),
        answer=answer,
    )
    # expected_entities is reference metadata → INFO, not PASS/FAIL.
    statuses = {r.field_name: r.status for r in result.results}
    assert statuses["expected_entities"] == "INFO"
    assert statuses["expected_entities_min_matches"] == "PASS"
    assert statuses["cross_source_facts_min_count"] == "PASS"


def test_cross_003_fails_when_bundle_misses_expected_entities():
    # Bundle has only one of the three expected CVEs.
    bundle = _cross_bundle(entities=("CVE-2023-22515",))
    answer = (
        "**Cross-Source Facts** (L2)\n"
        "- One fact [Source: NVD: CVE-2023-22515 | nvd_CVE-2023-22515].\n"
    )
    result = evaluate_field_coverage(
        "cross_003",
        _CROSS_003_REQ,
        template="CrossSourceCompare",
        fact_bundle=bundle,
        answer=answer,
    )
    entities_match = next(
        r for r in result.results if r.field_name == "expected_entities_min_matches"
    )
    facts = next(
        r for r in result.results if r.field_name == "cross_source_facts_min_count"
    )
    assert entities_match.status == "FAIL"
    assert facts.status == "FAIL"  # only 1 bullet, needs ≥ 2


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def test_aggregate_produces_per_template_and_per_layer_breakdown():
    r1 = evaluate_field_coverage(
        "vuln_004",
        _VULN_004_REQ,
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=(
            "**Mitigations** (L2)\n"
            "- Patch [Source: CISA KEV: CVE-2023-4966 | cisa_kev_CVE-2023-4966].\n"
        ),
    )
    r2 = evaluate_field_coverage(
        "ttp_001",
        _TTP_001_REQ,
        template="ThreatContext",
        fact_bundle=_threat_bundle(),
        answer=(
            "**Defensive Guidance** (L2)\n"
            "- Monitor [Source: MISP Event abc | misp_abc].\n"
        ),
    )
    agg = aggregate_field_coverage([r1, r2])
    assert agg["queries_evaluated"] == 2
    assert "VulnTriage" in agg["per_template"]
    assert "ThreatContext" in agg["per_template"]
    # Per-layer report should cover both L1 and L2.
    assert "L1" in agg["per_layer"]
    assert "L2" in agg["per_layer"]
    # Coverage ratios average to a sensible value in [0,1].
    assert 0.0 <= agg["mean_coverage_ratio"] <= 1.0


def test_evaluate_no_required_fields_returns_empty_result():
    result = evaluate_field_coverage(
        "vuln_005",
        {},
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer="",
    )
    assert result.results == []
    assert result.pass_count == 0
    assert result.coverage_ratio == 0.0


def test_evaluate_without_template_skips_all_fields():
    result = evaluate_field_coverage(
        "foo",
        _VULN_004_REQ,
        template=None,
        fact_bundle=None,
        answer="",
    )
    assert all(r.status == "SKIP" for r in result.results)
    assert result.fail_count == 0
    # Coverage ratio is 0 because nothing scored.
    assert result.coverage_ratio == 0.0


# ---------------------------------------------------------------------------
# Section parser edge cases
# ---------------------------------------------------------------------------


def test_l2_section_parser_ignores_content_before_header():
    answer = (
        "## Some Title\n\n"
        "- A bullet that should NOT be counted, outside any section.\n\n"
        "**Mitigations** (L2)\n"
        "- Real bullet [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n"
    )
    result = evaluate_field_coverage(
        "synthetic",
        {"mitigations_min_count": 1},
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=answer,
    )
    [r] = result.results
    assert r.status == "PASS"
    assert r.actual == 1


def test_l2_section_parser_stops_at_next_bold_header():
    answer = (
        "**Mitigations** (L2)\n"
        "- Patch [Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n\n"
        "**Evidence** (L1)\n"
        "- This is evidence, not a mitigation "
        "[Source: NVD: CVE-2023-4966 | nvd_CVE-2023-4966].\n"
    )
    result = evaluate_field_coverage(
        "synthetic",
        {"mitigations_min_count": 1},
        template="VulnTriage",
        fact_bundle=_vuln_bundle(),
        answer=answer,
    )
    [r] = result.results
    assert r.actual == 1, f"Evidence bullet leaked into Mitigations count: {r}"
