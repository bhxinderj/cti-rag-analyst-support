"""
Unit tests for the deterministic triage severity signal.

These tests anchor the rule table from docs/triage-template-spec.md §6 —
they are the canonical reference that the thesis methods section cites.
"""

from src.cti_rag.rag.severity import compute_triage_signal, SEVERITY_RULES


def test_critical_when_cvss_high_and_kev_listed():
    signal = compute_triage_signal(cvss_score=9.4, kev_listed=True, ransomware_use=False)
    assert signal.severity == "critical"
    assert signal.label == "CRITICAL — Immediate attention"
    assert "CVSS 9.4" in signal.rationale
    assert "active KEV listing" in signal.rationale


def test_critical_when_cvss_high_and_ransomware_use_without_kev():
    # Hypothetical branch: ransomware evidence is enough on its own to
    # push a 9.0+ CVE to critical even if the KEV flag is false.
    signal = compute_triage_signal(cvss_score=9.0, kev_listed=False, ransomware_use=True)
    assert signal.severity == "critical"


def test_high_prioritize_when_cvss_7_plus_and_kev_listed():
    # Rule 2 fires before rule 3: KEV-listed 7.5 ranks higher than a
    # non-KEV 9.0, because active exploitation trumps raw CVSS.
    signal = compute_triage_signal(cvss_score=7.5, kev_listed=True, ransomware_use=False)
    assert signal.severity == "high"
    assert signal.label == "HIGH — Prioritize"


def test_high_review_when_cvss_9_plus_without_kev():
    signal = compute_triage_signal(cvss_score=9.8, kev_listed=False, ransomware_use=False)
    assert signal.severity == "high"
    assert signal.label == "HIGH — Review"


def test_moderate_when_cvss_between_7_and_9_without_kev():
    signal = compute_triage_signal(cvss_score=7.5, kev_listed=False, ransomware_use=False)
    assert signal.severity == "moderate"
    assert signal.label == "MODERATE — Track"


def test_low_when_cvss_under_7():
    signal = compute_triage_signal(cvss_score=4.3, kev_listed=False, ransomware_use=False)
    assert signal.severity == "low"
    assert signal.label == "LOW — Routine"


def test_unknown_when_cvss_unavailable():
    signal = compute_triage_signal(cvss_score=None, kev_listed=True, ransomware_use=True)
    assert signal.severity == "unknown"
    assert "unavailable" in signal.rationale.lower()


def test_rule_boundaries():
    """Tabular sweep of boundary cases — one assertion per row."""
    cases = [
        (9.0, True, False, "critical"),  # boundary, KEV
        (9.0, False, True, "critical"),  # boundary, ransomware
        (8.9, True, False, "high"),  # just below 9, but KEV
        (7.0, True, False, "high"),  # boundary for rule 2
        (6.9, True, False, "low"),  # KEV but below CVSS 7
        (9.0, False, False, "high"),  # critical-high without exploitation
        (7.0, False, False, "moderate"),  # boundary moderate
        (6.99, False, False, "low"),  # boundary low
    ]
    for cvss, kev, rans, expected in cases:
        actual = compute_triage_signal(cvss, kev, rans).severity
        assert actual == expected, (
            f"cvss={cvss}, kev={kev}, ransomware={rans}: "
            f"expected {expected!r}, got {actual!r}"
        )


def test_signal_to_dict_is_json_serializable():
    import json

    signal = compute_triage_signal(cvss_score=9.4, kev_listed=True, ransomware_use=True)
    payload = signal.to_dict()
    # Round-trip through json.dumps to confirm no non-JSON values snuck in.
    assert json.loads(json.dumps(payload)) == payload
    assert payload["inputs"]["cvss"] == 9.4
    assert payload["inputs"]["kev_listed"] is True


def test_severity_rules_constant_matches_expected_shape():
    # Sanity check on the module-level documentation table.
    assert len(SEVERITY_RULES) == 6
    assert {rule["severity"] for rule in SEVERITY_RULES} == {
        "critical",
        "high",
        "moderate",
        "low",
        "unknown",
    }


def test_inputs_do_not_silently_coerce_none_kev_to_false_surprise():
    # Passing non-bool truthy/falsy values should still land in the
    # right bucket and not crash.
    signal = compute_triage_signal(cvss_score=9.4, kev_listed=1, ransomware_use=0)  # type: ignore[arg-type]
    assert signal.severity == "critical"
    assert signal.inputs["kev_listed"] is True
    assert signal.inputs["ransomware_use"] is False
