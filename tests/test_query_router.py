"""
Unit tests for the deterministic query router.

The router is the entry point of Phase 2 — it picks which of the three
triage templates is rendered. Routing logic lives in a single precedence
table, so the tests mirror that: one targeted test per rule, then a
coverage sweep against every query id in ``configs/eval_queries.yaml``.
"""

from pathlib import Path

import yaml

from src.cti_rag.rag.router import classify_query


# ---------------------------------------------------------------------------
# Per-rule tests
# ---------------------------------------------------------------------------


def test_rule_1_two_source_names_route_to_cross_source_compare():
    decision = classify_query(
        "Which CVEs appear in both NVD and CISA KEV with confirmed ransomware use?"
    )
    assert decision.template == "CrossSourceCompare"
    assert decision.rule_matched == 1
    assert set(decision.matched_sources) == {"nvd", "cisa_kev"}


def test_rule_1_does_not_double_count_cisa_kev_as_cisa_plus_kev():
    # Only CISA KEV is named — the bare "CISA" pattern must not fire
    # a second time. Router should fall through to rule 3 (CVE).
    decision = classify_query("Is CVE-2023-4966 listed in the CISA KEV catalog?")
    assert decision.template == "VulnTriage"
    assert decision.matched_sources == ["cisa_kev"]


def test_rule_2_compare_keyword_routes_to_cross_source_compare():
    decision = classify_query(
        "Compare the severity of recent Microsoft Exchange Server vulnerabilities."
    )
    assert decision.template == "CrossSourceCompare"
    assert decision.rule_matched == 2


def test_rule_2_versus_keyword_routes_to_cross_source_compare():
    decision = classify_query("CVE-2021-44228 vs. CVE-2021-45046")
    # "vs." triggers rule 2 before rule 3 (CVE pattern). This is
    # intentional: vs/compare phrasing overrides pure CVE mention.
    assert decision.template == "CrossSourceCompare"
    assert decision.rule_matched == 2


def test_rule_3_cve_id_routes_to_vuln_triage():
    decision = classify_query(
        "What is CVE-2023-4966 (Citrix Bleed) and how severe is it?"
    )
    assert decision.template == "VulnTriage"
    assert decision.rule_matched == 3
    assert decision.primary_entities == ["CVE-2023-4966"]


def test_rule_3_extracts_multiple_cve_ids_in_order():
    decision = classify_query(
        "Describe CVE-2023-46805 and CVE-2024-21887 and their combined impact."
    )
    assert decision.template == "VulnTriage"
    assert decision.primary_entities == ["CVE-2023-46805", "CVE-2024-21887"]


def test_rule_4_technique_id_routes_to_threat_context():
    decision = classify_query(
        "What phishing-related techniques (T1566) are documented?"
    )
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 4
    assert decision.threat_context_kind == "technique"
    assert decision.primary_entities == ["T1566"]


def test_rule_4_tactic_id_routes_to_threat_context():
    decision = classify_query("Which techniques relate to TA0004?")
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 4
    assert decision.threat_context_kind == "tactic"
    assert decision.primary_entities == ["TA0004"]


def test_rule_5_ipv4_routes_to_threat_context():
    decision = classify_query("Is 198.51.100.10 a known IoC?")
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 5
    assert decision.threat_context_kind == "ioc"
    assert "198.51.100.10" in decision.primary_entities


def test_rule_5_sha256_routes_to_threat_context():
    sha = "a" * 64
    decision = classify_query(f"Any threat reports mentioning hash {sha}?")
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 5


def test_rule_6_mitre_attack_keyword_without_id_routes_to_threat_context():
    decision = classify_query(
        "What initial access techniques (MITRE ATT&CK) relate to public-facing apps?"
    )
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 6
    # "initial access" tactic keyword resolves to TA0001.
    assert decision.primary_entities == ["TA0001"]


def test_rule_6_tactic_name_alone_routes_to_threat_context():
    decision = classify_query(
        "Describe persistence mechanisms via remote services."
    )
    assert decision.template == "ThreatContext"
    assert decision.rule_matched == 6
    assert decision.primary_entities == ["TA0003"]


def test_rule_7_fallback_to_vuln_triage():
    decision = classify_query(
        "Which critical vulnerabilities affect Apache Log4j?"
    )
    assert decision.template == "VulnTriage"
    assert decision.rule_matched == 7
    assert decision.primary_entities == []


# ---------------------------------------------------------------------------
# Precedence tests
# ---------------------------------------------------------------------------


def test_rule_1_beats_rule_3_when_both_sources_and_cve_present():
    decision = classify_query(
        "For CVE-2023-4966, summarize what NVD and CISA KEV report."
    )
    assert decision.template == "CrossSourceCompare"
    assert decision.rule_matched == 1


def test_rule_3_beats_rule_6_when_both_cve_and_tactic_keyword_present():
    # "impact" keyword could match rule 6, but CVE pattern fires first.
    decision = classify_query("What is the impact of CVE-2024-3094?")
    assert decision.template == "VulnTriage"
    assert decision.rule_matched == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_query_returns_fallback_vuln_triage():
    assert classify_query("").rule_matched == 7
    assert classify_query("   ").rule_matched == 7


def test_decision_as_trace_dict_is_json_ready():
    import json

    decision = classify_query("What is CVE-2023-4966?")
    payload = decision.as_trace_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["template"] == "VulnTriage"
    assert payload["primary_entities"] == ["CVE-2023-4966"]


# ---------------------------------------------------------------------------
# Coverage sweep over the full eval set
# ---------------------------------------------------------------------------


def test_router_classifies_all_28_eval_queries_defensibly():
    """
    Every query in ``configs/eval_queries.yaml`` must classify to one of
    the three known templates, and the three calibration cases must
    match their ``template_expected`` annotation exactly.
    """
    queries_path = Path("configs/eval_queries.yaml")
    data = yaml.safe_load(queries_path.read_text())
    queries = data["queries"]

    assert len(queries) == 28, "unexpected number of eval queries"

    calibration_mismatches = []
    for q in queries:
        decision = classify_query(q["question"])
        assert decision.template in {
            "VulnTriage",
            "ThreatContext",
            "CrossSourceCompare",
        }, f"{q['id']} routed to unexpected template {decision.template!r}"

        expected = q.get("template_expected")
        if expected and expected != decision.template:
            calibration_mismatches.append((q["id"], expected, decision.template))

    assert not calibration_mismatches, (
        "calibration-case routing mismatches: " + repr(calibration_mismatches)
    )
