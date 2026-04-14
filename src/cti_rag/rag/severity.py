"""
Deterministic severity signal ("triage ampel") for the VulnTriage template.

This module implements the rule table from docs/triage-template-spec.md
§6. It is intentionally a pure function over three inputs — CVSS base
score, KEV listing, and known-ransomware-use flag — so the thesis can
describe it as a table and defend it as a regelbased design choice that
replaces an LLM judgement with a deterministic signal.

The rules in priority order (first match wins)
----------------------------------------------
1. CRITICAL — Immediate attention
       CVSS ≥ 9.0  AND  (KEV listed OR known ransomware use)
2. HIGH — Prioritize
       CVSS ≥ 7.0  AND  KEV listed
3. HIGH — Review
       CVSS ≥ 9.0  AND  NOT KEV listed
4. MODERATE — Track
       7.0 ≤ CVSS < 9.0  AND  NOT KEV listed
5. LOW — Routine
       CVSS < 7.0  (only when CVSS is known)
6. UNKNOWN — Manual review required
       CVSS unavailable

Note that rule 2 fires before rule 3 on purpose: an actively exploited
CVE at CVSS 7.5 is a higher triage priority than a non-KEV CVE at
CVSS 9.0 that has no evidence of exploitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Exposed as a module-level constant so tests, the thesis methods
# section, and any downstream tooling can reference the exact rule table
# without re-implementing it.
SEVERITY_RULES: tuple[dict[str, Any], ...] = (
    {
        "severity": "critical",
        "label": "CRITICAL — Immediate attention",
        "predicate": "cvss >= 9.0 AND (kev_listed OR ransomware_use)",
    },
    {
        "severity": "high",
        "label": "HIGH — Prioritize",
        "predicate": "cvss >= 7.0 AND kev_listed",
    },
    {
        "severity": "high",
        "label": "HIGH — Review",
        "predicate": "cvss >= 9.0 AND NOT kev_listed",
    },
    {
        "severity": "moderate",
        "label": "MODERATE — Track",
        "predicate": "7.0 <= cvss < 9.0 AND NOT kev_listed",
    },
    {
        "severity": "low",
        "label": "LOW — Routine",
        "predicate": "cvss < 7.0",
    },
    {
        "severity": "unknown",
        "label": "UNKNOWN — Manual review required",
        "predicate": "cvss unavailable",
    },
)


@dataclass(frozen=True)
class TriageSignal:
    """Deterministic severity assessment attached to a VulnTriage bundle."""

    severity: str  # machine key: critical | high | moderate | low | unknown
    label: str  # human-readable display
    rationale: str  # short natural-language explanation
    inputs: dict[str, Any]  # echo of (cvss, kev_listed, ransomware_use)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form suitable for JSON serialization."""
        return asdict(self)


def _format_rationale(cvss: float | None, kev_listed: bool, ransomware_use: bool) -> str:
    parts: list[str] = []
    if cvss is not None:
        parts.append(f"CVSS {cvss:g}")
    else:
        parts.append("CVSS unavailable")
    if kev_listed:
        parts.append("active KEV listing")
    if ransomware_use:
        parts.append("known ransomware use")
    if not kev_listed and not ransomware_use and cvss is not None:
        parts.append("no KEV listing, no ransomware evidence")
    return " + ".join(parts)


def compute_triage_signal(
    cvss_score: float | None,
    kev_listed: bool,
    ransomware_use: bool,
) -> TriageSignal:
    """
    Apply the rule table above and return a :class:`TriageSignal`.

    ``cvss_score`` is treated as unavailable when ``None``. Callers must
    pass booleans for the remaining two flags — the function does not
    coerce, so that a missing KEV metadata field does not silently slide
    into "listed".
    """
    kev = bool(kev_listed)
    ransomware = bool(ransomware_use)
    inputs: dict[str, Any] = {
        "cvss": cvss_score,
        "kev_listed": kev,
        "ransomware_use": ransomware,
    }
    rationale = _format_rationale(cvss_score, kev, ransomware)

    if cvss_score is None:
        return TriageSignal(
            severity="unknown",
            label="UNKNOWN — Manual review required",
            rationale=rationale,
            inputs=inputs,
        )

    if cvss_score >= 9.0 and (kev or ransomware):
        return TriageSignal(
            severity="critical",
            label="CRITICAL — Immediate attention",
            rationale=rationale,
            inputs=inputs,
        )

    if cvss_score >= 7.0 and kev:
        return TriageSignal(
            severity="high",
            label="HIGH — Prioritize",
            rationale=rationale,
            inputs=inputs,
        )

    if cvss_score >= 9.0:  # implies NOT kev by fall-through
        return TriageSignal(
            severity="high",
            label="HIGH — Review",
            rationale=rationale,
            inputs=inputs,
        )

    if 7.0 <= cvss_score < 9.0:  # implies NOT kev by fall-through
        return TriageSignal(
            severity="moderate",
            label="MODERATE — Track",
            rationale=rationale,
            inputs=inputs,
        )

    return TriageSignal(
        severity="low",
        label="LOW — Routine",
        rationale=rationale,
        inputs=inputs,
    )


def annotate_vuln_triage_bundle(facts) -> None:
    """
    Attach a :class:`TriageSignal` (as dict) to a ``VulnTriageFacts``.

    Kept here rather than in ``facts.py`` to avoid a circular import and
    to keep the severity logic in a single, thesis-citable module.
    Operates in-place; returns ``None``.
    """
    signal = compute_triage_signal(
        cvss_score=getattr(facts, "cvss_score", None),
        kev_listed=bool(getattr(facts, "kev_listed", False)),
        ransomware_use=bool(getattr(facts, "ransomware_use", False)),
    )
    # Store as plain dict so downstream consumers (template renderer,
    # evaluation) can treat the field as JSON-serializable without
    # importing the TriageSignal dataclass.
    facts.triage_signal = signal.to_dict()
