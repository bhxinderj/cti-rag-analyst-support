"""
Unit tests for the three Phase-2 template renderers and prompt builders.

These tests exercise the deterministic L1 output (what the analyst sees
verbatim in the final answer) and the shape of the LLM prompt — we do
NOT invoke the LLM here. Prompt content is checked for the invariants
the prompt must carry: citation label allowlist, the pre-rendered L1
block, and explicit section targets for the model.
"""

import json

from src.cti_rag.rag.facts import build_fact_bundle
from src.cti_rag.rag.severity import annotate_vuln_triage_bundle
from src.cti_rag.rag.templates import (
    build_cross_source_prompt,
    build_prompt,
    build_threat_context_prompt,
    build_vuln_triage_prompt,
    render_cross_source_l1,
    render_l1,
    render_threat_context_l1,
    render_vuln_triage_l1,
)


# ---------------------------------------------------------------------------
# Shared fixtures (synthetic chunks)
# ---------------------------------------------------------------------------


def _nvd_chunk(cve_id="CVE-2023-4966"):
    return {
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
        },
    }


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
            "required_action": "Apply vendor patch per Citrix advisory.",
            "known_ransomware": "Known" if ransomware else "Unknown",
            "known_ransomware_bool": ransomware,
        },
    }


def _misp_chunk(misp_id="abc-123", cve_ids="CVE-2024-24919", technique="T1190"):
    iocs = [
        {"type": "ip-dst", "value": "198.51.100.10", "category": None},
        {"type": "domain", "value": "malicious.example.org", "category": None},
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
        },
    }


# ---------------------------------------------------------------------------
# VulnTriage
# ---------------------------------------------------------------------------


def test_render_vuln_triage_l1_contains_all_deterministic_fields():
    chunks = [_nvd_chunk(), _kev_chunk()]
    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks)

    rendered = render_vuln_triage_l1(bundle)

    # Header + CVE present
    assert "CVE-2023-4966" in rendered
    # Triage signal is rendered without the LLM being involved
    assert "Triage Signal:" in rendered
    assert "CRITICAL" in rendered
    # Deterministic fields
    assert "CVSS: 9.4" in rendered
    assert "CWE-119" in rendered
    assert "Citrix NetScaler ADC" in rendered
    assert "Citrix NetScaler Gateway" in rendered
    # KEV block
    assert "Listed in CISA KEV: yes" in rendered
    assert "2023-10-18" in rendered
    assert "2023-11-08" in rendered
    assert "Known ransomware use: yes" in rendered


def test_render_vuln_triage_l1_handles_missing_cvss_and_kev_cleanly():
    # Only a KEV-ish chunk without CVSS metadata.
    chunk = _kev_chunk(ransomware=False)
    chunk["metadata"].pop("known_ransomware_bool", None)
    chunk["metadata"]["known_ransomware"] = "Unknown"

    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], [chunk])
    rendered = render_vuln_triage_l1(bundle)

    assert "CVSS: not available in retrieved context" in rendered
    # Severity signal still rendered as "UNKNOWN".
    assert "UNKNOWN" in rendered


def test_render_vuln_triage_l1_renders_multiple_cards_separated_by_hr():
    chunks = [_nvd_chunk("CVE-2023-46805"), _nvd_chunk("CVE-2024-21887")]
    bundle = build_fact_bundle(
        "VulnTriage", ["CVE-2023-46805", "CVE-2024-21887"], chunks
    )

    rendered = render_vuln_triage_l1(bundle)
    assert "CVE-2023-46805" in rendered
    assert "CVE-2024-21887" in rendered
    # The two cards are separated by a horizontal rule line.
    assert "\n---\n" in rendered


def test_render_vuln_triage_l1_surfaces_cvss_inconsistency():
    alt = _nvd_chunk()
    alt["metadata"]["cvss_score"] = 7.5
    alt["doc_id"] = "nvd_alt_CVE-2023-4966"
    alt["title"] = "NVD alt: CVE-2023-4966"

    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], [_nvd_chunk(), alt])
    rendered = render_vuln_triage_l1(bundle)

    assert "Source Inconsistencies" in rendered
    assert "CVSS" in rendered


def test_build_vuln_triage_prompt_carries_l1_block_and_citation_allowlist():
    chunks = [_nvd_chunk(), _kev_chunk()]
    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks)

    prompt = build_vuln_triage_prompt(
        bundle, "What is CVE-2023-4966?", chunks
    )
    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"

    user = prompt[1]["content"]
    # The pre-rendered L1 block is in the user prompt.
    assert "Pre-rendered L1 Block" in user
    assert "CRITICAL" in user
    # Allowed citation labels include both chunks.
    assert "nvd_CVE-2023-4966" in user
    assert "cisa_kev_CVE-2023-4966" in user
    # Target sections requested, L1 block section NOT re-asked.
    assert "Exploitation Context" in user
    assert "Mitigations" in user
    assert "Evidence" in user
    assert "Gaps" in user


# ---------------------------------------------------------------------------
# ThreatContext
# ---------------------------------------------------------------------------


def test_render_threat_context_l1_for_technique_includes_tactic_lookup():
    chunks = [_misp_chunk(technique="T1190")]
    bundle = build_fact_bundle(
        "ThreatContext",
        ["T1190"],
        chunks,
        threat_context_kind="technique",
    )
    rendered = render_threat_context_l1(bundle)

    assert "T1190" in rendered
    assert "Associated CVEs" in rendered
    assert "CVE-2024-24919" in rendered
    assert "IoC Summary" in rendered
    assert "IP addresses: 1 observed" in rendered
    assert "Domains: 1 observed" in rendered


def test_render_threat_context_l1_for_tactic_resolves_tactic_name():
    bundle = build_fact_bundle(
        "ThreatContext", ["TA0001"], [], threat_context_kind="tactic"
    )
    rendered = render_threat_context_l1(bundle)

    assert "TA0001" in rendered
    assert "Initial Access" in rendered


def test_build_threat_context_prompt_carries_primary_entity_and_targets_l2_sections():
    chunks = [_misp_chunk(technique="T1190")]
    bundle = build_fact_bundle(
        "ThreatContext",
        ["T1190"],
        chunks,
        threat_context_kind="technique",
    )

    prompt = build_threat_context_prompt(
        bundle, "What maps to T1190?", chunks
    )
    user = prompt[1]["content"]
    assert "Primary entity: T1190 (technique)" in user
    assert "Threat Actors and Malware" in user
    assert "Defensive Guidance" in user


# ---------------------------------------------------------------------------
# CrossSourceCompare
# ---------------------------------------------------------------------------


def test_render_cross_source_l1_renders_comparison_table_and_top_matches():
    chunks = [
        _nvd_chunk("CVE-2023-22515"),
        _kev_chunk("CVE-2023-22515", ransomware=False),
        _nvd_chunk("CVE-2023-34362"),
        _kev_chunk("CVE-2023-34362"),
    ]
    bundle = build_fact_bundle(
        "CrossSourceCompare",
        ["CVE-2023-22515", "CVE-2023-34362"],
        chunks,
    )

    rendered = render_cross_source_l1(bundle)

    # Summary counts
    assert "Entities evaluated: 2" in rendered
    assert "Entities with data from NVD and CISA KEV: 2" in rendered
    # Comparison table header + rows
    assert "| CVE | CVSS |" in rendered
    assert "CVE-2023-22515" in rendered
    assert "CVE-2023-34362" in rendered
    # Triage signal annotation in top-matches
    assert "CRITICAL" in rendered
    # Ransomware column values
    assert "| yes |" in rendered or "yes |" in rendered


def test_render_cross_source_l1_handles_missing_entity_explicitly():
    # Request two entities; retrieve only one.
    chunks = [_nvd_chunk("CVE-2023-22515"), _kev_chunk("CVE-2023-22515")]
    bundle = build_fact_bundle(
        "CrossSourceCompare",
        ["CVE-2023-22515", "CVE-2099-00000"],
        chunks,
    )
    rendered = render_cross_source_l1(bundle)

    assert "Entities requested but not present in retrieved context: 1" in rendered
    assert "CVE-2099-00000" in rendered


def test_build_cross_source_prompt_asks_for_source_agreement_section():
    chunks = [_nvd_chunk("CVE-2023-22515"), _kev_chunk("CVE-2023-22515")]
    bundle = build_fact_bundle(
        "CrossSourceCompare", ["CVE-2023-22515"], chunks
    )
    prompt = build_cross_source_prompt(
        bundle, "Compare sources for CVE-2023-22515.", chunks
    )
    user = prompt[1]["content"]

    assert "Source Agreement" in user
    assert "Cross-Source Facts" in user
    assert "CVE-2023-22515" in user


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_render_l1_dispatch_matches_template():
    chunks = [_nvd_chunk(), _kev_chunk()]
    vuln_bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks)
    tc_bundle = build_fact_bundle(
        "ThreatContext",
        ["T1190"],
        [_misp_chunk()],
        threat_context_kind="technique",
    )
    cs_bundle = build_fact_bundle(
        "CrossSourceCompare", ["CVE-2023-4966"], chunks
    )

    assert render_l1(vuln_bundle) == render_vuln_triage_l1(vuln_bundle)
    assert render_l1(tc_bundle) == render_threat_context_l1(tc_bundle)
    assert render_l1(cs_bundle) == render_cross_source_l1(cs_bundle)


def test_build_prompt_dispatch_returns_two_message_list_per_template():
    chunks = [_nvd_chunk(), _kev_chunk()]
    bundles = [
        build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks),
        build_fact_bundle(
            "ThreatContext",
            ["T1190"],
            [_misp_chunk()],
            threat_context_kind="technique",
        ),
        build_fact_bundle("CrossSourceCompare", ["CVE-2023-4966"], chunks),
    ]
    for bundle in bundles:
        prompt = build_prompt(bundle, "q", chunks)
        assert len(prompt) == 2
        assert {m["role"] for m in prompt} == {"system", "user"}


def test_l1_block_never_repeats_instructions_or_placeholders():
    # Guardrail: the L1 block should not contain instructions meant for
    # the LLM. Catches regressions where a template string leaks into
    # the deterministic block.
    chunks = [_nvd_chunk(), _kev_chunk()]
    bundle = build_fact_bundle("VulnTriage", ["CVE-2023-4966"], chunks)
    annotate_vuln_triage_bundle(bundle.vuln_triage[0])
    rendered = render_vuln_triage_l1(bundle)

    forbidden = ["TODO", "{placeholder}", "do NOT repeat", "Produce ONLY"]
    for token in forbidden:
        assert token not in rendered, f"L1 block leaked prompt token {token!r}"
