"""
Triage Field Coverage evaluator (Phase 2, spec §8).

Role in the pipeline
--------------------
Given a :class:`RAGResponse` from ``RAGChain.query_templated`` and the
``required_fields`` annotation from ``configs/eval_queries.yaml``, this
module decides — **per field** — whether the deterministic pipeline met
the analyst's expectation.

Two kinds of fields
-------------------
**Layer-1 fields** are read directly from the serialized ``FactBundle``
attached to the ``RAGResponse``. They never depend on LLM output and
cannot be hallucinated — this is the spec's core design move (§ 8.1).

**Layer-2 fields** (suffix ``_min_count`` for ``mitigations``,
``detection_or_mitigation``, ``cross_source_facts``) describe content the
LLM is asked to generate. They are measured by counting grounded bullets
inside the corresponding bold-markdown section of the rendered answer.

See docs/triage-template-spec.md §§ 8.1, 8.2 for the contract.

Scope note
----------
Per §9.1/§9.2, only three calibrated annotations exist at Phase-2 freeze
(``vuln_004``, ``ttp_001``, ``cross_003``). This evaluator is designed
to handle arbitrary additional annotations of the same shape — when the
remaining 25 queries are annotated in Phase 3 (§11), no code change here
should be required.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..rag.facts import extract_attack_techniques

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """Single-field evaluation outcome."""

    field_name: str
    status: str  # "PASS" | "FAIL" | "SKIP" | "INFO"
    expected: Any
    actual: Any
    detail: str = ""
    layer: str = "L1"  # "L1" (FactBundle) | "L2" (rendered answer) | "meta"


@dataclass
class FieldCoverageResult:
    """Aggregated per-query coverage outcome."""

    query_id: str
    template: str | None
    results: list[FieldResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.results if r.status == "SKIP")

    @property
    def info_count(self) -> int:
        return sum(1 for r in self.results if r.status == "INFO")

    @property
    def scored_count(self) -> int:
        """Fields that counted toward coverage (PASS + FAIL)."""
        return self.pass_count + self.fail_count

    @property
    def coverage_ratio(self) -> float:
        """``filled_correctly / required_fields`` per spec §8.2."""
        total = self.scored_count
        return (self.pass_count / total) if total else 0.0

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "template": self.template,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "info_count": self.info_count,
            "scored_count": self.scored_count,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "fields": [
                {
                    "field": r.field_name,
                    "status": r.status,
                    "layer": r.layer,
                    "expected": r.expected,
                    "actual": r.actual,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Matchers (spec §8.2 "Variante Y")
# ---------------------------------------------------------------------------


def _match_numeric(expected: float, actual: Any, tol: float = 0.1) -> tuple[bool, str]:
    if actual is None:
        return False, "actual is None"
    try:
        return (
            abs(float(actual) - float(expected)) <= tol,
            f"actual={actual}, expected={expected} (tol {tol})",
        )
    except (TypeError, ValueError):
        return False, f"actual={actual!r} not numeric"


def _match_bool(expected: bool, actual: Any) -> tuple[bool, str]:
    return bool(actual) == bool(expected), f"actual={actual}, expected={expected}"


def _match_exact_ci(expected: str, actual: Any) -> tuple[bool, str]:
    if actual is None:
        return False, "actual is None"
    return (
        str(actual).strip().lower() == str(expected).strip().lower(),
        f"actual={actual!r}, expected={expected!r}",
    )


def _match_list_any(expected: list, actual: Any) -> tuple[bool, str]:
    """any_substring: at least one expected item appears in actual list."""
    if not actual:
        return False, "actual is empty"
    if not isinstance(actual, (list, tuple, set)):
        actual = [actual]
    actual_norm = [str(x).strip().lower() for x in actual]
    for exp in expected:
        exp_norm = str(exp).strip().lower()
        for actual_item in actual_norm:
            if exp_norm == actual_item or exp_norm in actual_item or actual_item in exp_norm:
                return True, f"matched {exp!r} in {actual}"
    return False, f"none of {expected} in {actual}"


def _match_min_count(expected: int, actual_count: int) -> tuple[bool, str]:
    return (
        int(actual_count) >= int(expected),
        f"count={actual_count}, required ≥ {expected}",
    )


def _match_min_matches(
    expected_list: list, actual_list: list, required: int
) -> tuple[bool, str]:
    actual_norm = {str(x).strip().upper() for x in actual_list or []}
    hits = [e for e in expected_list if str(e).strip().upper() in actual_norm]
    return (
        len(hits) >= int(required),
        f"matched {len(hits)}/{len(expected_list)}, required ≥ {required}: {hits}",
    )


# ---------------------------------------------------------------------------
# L1 field extractors — read from serialized FactBundle (dict form)
#
# The ``RAGChain.query_templated`` path attaches the FactBundle to
# ``RAGResponse.fact_bundle`` via ``dataclasses.asdict``. We read that
# dict here rather than reimporting the dataclass types, so the evaluator
# also works against JSON-serialized responses loaded from disk.
# ---------------------------------------------------------------------------


def _l1_vuln_triage(bundle: dict, field_name: str, _answer: str = "") -> Any:
    entities = bundle.get("vuln_triage") or []
    if not entities:
        return None
    primary = entities[0]
    triage = primary.get("triage_signal") or {}
    mapping: dict[str, Any] = {
        "cve_id": primary.get("cve_id"),
        "cvss_score": primary.get("cvss_score"),
        "cwe_ids": primary.get("cwe_ids") or [],
        "affected_products": primary.get("affected_products") or [],
        "kev_listed": primary.get("kev_listed"),
        "ransomware_use": primary.get("ransomware_use"),
        "triage_signal": triage.get("severity"),
    }
    return mapping.get(field_name, "__UNKNOWN__")


def _l1_threat_context(bundle: dict, field_name: str, _answer: str = "") -> Any:
    tc = bundle.get("threat_context") or {}
    if not tc:
        return None
    # ``attack_techniques`` aggregates tc.attack_technique + anything the
    # retrieved chunks carry in ``metadata.attack_techniques`` (comma-joined)
    # and in the chunk text. The evaluator needs the same aggregation the
    # L1 smoke used — but we only have the bundle here, so for tactic-kind
    # queries we fall back to the ``associated_cves`` aggregate plus the
    # primary entity. Better: the L1 renderer exposes nothing else, so
    # aggregated techniques are read out of ``supporting_chunks`` /
    # ``source_chunks`` of the bundle serialization.
    aggregated: list[str] = []
    if tc.get("attack_technique"):
        aggregated.append(str(tc["attack_technique"]).upper())

    # Pull from every chunk referenced in the bundle: fact bundles carry
    # source_chunks (per-source metadata summary) — but not the full
    # metadata. So the evaluator accepts an additional ``chunks`` hook
    # through ``_evaluate_one_field``. When not supplied, we can only use
    # what tc exposes directly.
    if field_name == "attack_techniques":
        return aggregated

    if field_name == "attack_tactic":
        parts: list[str] = []
        if tc.get("tactic_id"):
            parts.append(tc["tactic_id"])
        if tc.get("tactic_name"):
            parts.append(tc["tactic_name"])
        return parts

    if field_name == "associated_cves_min_count":
        return len(tc.get("associated_cves") or [])

    return "__UNKNOWN__"


def _l1_cross_source(bundle: dict, field_name: str, _answer: str = "") -> Any:
    cs = bundle.get("cross_source") or {}
    if not cs:
        return None
    entities = cs.get("entities") or []
    if field_name == "expected_entities":
        return [e.get("cve_id") for e in entities]
    return "__UNKNOWN__"


# ---------------------------------------------------------------------------
# ThreatContext technique aggregation — needs access to raw chunk metadata.
# The fact bundle does not carry ``attack_techniques`` comma-strings; the
# run_l1_smoke helper walked the chunks directly. We replicate that here
# when ``source_documents`` is supplied alongside the bundle.
# ---------------------------------------------------------------------------


def _aggregate_attack_techniques(
    tc: dict, source_documents: list[dict] | None
) -> list[str]:
    aggregated: list[str] = []
    if tc.get("attack_technique"):
        aggregated.append(str(tc["attack_technique"]).upper())
    if not source_documents:
        return aggregated
    for chunk in source_documents:
        meta = chunk.get("metadata") or {}
        raw = meta.get("attack_techniques") or ""
        if isinstance(raw, str):
            for tid in raw.split(","):
                tid = tid.strip().upper()
                if tid and tid not in aggregated:
                    aggregated.append(tid)
        for tid in extract_attack_techniques(
            (chunk.get("title") or "") + " " + (chunk.get("content") or "")
        ):
            if tid not in aggregated:
                aggregated.append(tid)
    return aggregated


# ---------------------------------------------------------------------------
# L2 bullet counting — parse rendered answer for grounded bullets in a
# specific bold-markdown section. Sections are delimited by literal
# ``**Section Name**`` tokens on their own line (or start-of-line) and
# end at the next ``**...**`` header, a horizontal rule, or EOF.
#
# Template fallback phrases ("No reliable mitigation guidance is present
# in the retrieved context.") are treated as count=0 rather than a bullet.
# ---------------------------------------------------------------------------


_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_BOLD_HEADER_RE = re.compile(r"^\s*\*\*([^*]+?)\*\*\s*(?:\([LL][12]\))?\s*$")
_HR_RE = re.compile(r"^\s*-{3,}\s*$")

# The template instructs the LLM to use these exact phrases for empty
# sections. We must not count them as bullets.
_EMPTY_FALLBACKS = {
    "no reliable mitigation guidance is present in the retrieved context.",
    "no grounded information in the retrieved context.",
    "no defensive guidance present in the retrieved context.",
}


def _extract_section_body(answer: str, section_title: str) -> list[str]:
    """Return the non-empty body lines of ``**section_title**`` in ``answer``.

    Accepts case-insensitive matches and tolerates trailing ``(L2)`` /
    ``(L1)`` annotations on the header line (Phase-2 templates use this).
    Returns ``[]`` when the section is absent.
    """
    target = section_title.strip().lower()
    lines = answer.splitlines()

    # Locate header line.
    start: int | None = None
    for i, raw in enumerate(lines):
        m = _BOLD_HEADER_RE.match(raw)
        if m and m.group(1).strip().lower() == target:
            start = i + 1
            break
    if start is None:
        return []

    body: list[str] = []
    for raw in lines[start:]:
        if _BOLD_HEADER_RE.match(raw):
            break
        if _HR_RE.match(raw):
            break
        body.append(raw)

    # Strip leading/trailing blank lines.
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return body


def _count_grounded_bullets(body_lines: list[str]) -> int:
    """Count bullets that carry a ``[Source: ...]`` citation.

    Bullet lines without a citation are NOT counted — the field-coverage
    metric rewards grounded claims, not raw bullet density. An inline
    continuation line (indented under a bullet) is treated as part of the
    prior bullet for citation purposes.
    """
    if not body_lines:
        return 0

    # Collapse continuation lines into their owning bullet.
    collapsed: list[str] = []
    for raw in body_lines:
        stripped_lower = raw.strip().lower()
        if any(stripped_lower == fb or stripped_lower.startswith(fb) for fb in _EMPTY_FALLBACKS):
            # Explicit "no content" fallback — zero grounded bullets.
            return 0
        if _BULLET_PREFIX_RE.match(raw):
            collapsed.append(raw)
        elif collapsed and raw.strip():
            # Continuation: append to the previous bullet so a citation on
            # the continuation still counts toward the bullet above.
            collapsed[-1] = collapsed[-1] + " " + raw.strip()
        # else: blank or loose text — ignore for counting.

    grounded = 0
    for bullet in collapsed:
        if "[Source:" in bullet or "[source:" in bullet.lower():
            grounded += 1
    return grounded


# Mapping: L2 field name → list of candidate section titles to inspect
# (first match wins). The CTI-RAG templates use slightly different
# wording across the three templates, so we accept all plausible names.
_L2_SECTION_TITLES: dict[str, tuple[str, ...]] = {
    "mitigations_min_count": ("Mitigations", "Recommended actions / Mitigations"),
    "detection_or_mitigation_min_count": (
        "Defensive Guidance",
        "Mitigations",
        "Recommended actions / Mitigations",
    ),
    "cross_source_facts_min_count": ("Cross-Source Facts",),
}


def _l2_min_count(answer: str, field_name: str) -> int:
    """Count grounded bullets in the relevant L2 section of the answer."""
    titles = _L2_SECTION_TITLES.get(field_name, ())
    for title in titles:
        body = _extract_section_body(answer, title)
        if body:
            return _count_grounded_bullets(body)
    return 0


# ---------------------------------------------------------------------------
# Per-template dispatcher
# ---------------------------------------------------------------------------


_L1_DISPATCH: dict[str, Callable[[dict, str, str], Any]] = {
    "VulnTriage": _l1_vuln_triage,
    "ThreatContext": _l1_threat_context,
    "CrossSourceCompare": _l1_cross_source,
}


def _evaluate_one_field(
    template: str,
    field_name: str,
    expected: Any,
    bundle: dict,
    answer: str,
    source_documents: list[dict] | None,
    all_required: dict,
) -> FieldResult:
    """Route one field to its matcher, returning a :class:`FieldResult`."""
    # ------------------------------------------------------------------ L2
    if field_name in _L2_SECTION_TITLES:
        actual = _l2_min_count(answer, field_name)
        ok, detail = _match_min_count(int(expected), actual)
        return FieldResult(
            field_name=field_name,
            status="PASS" if ok else "FAIL",
            expected=expected,
            actual=actual,
            detail=detail,
            layer="L2",
        )

    # --------------------------------------------------- CrossSourceCompare
    # Special cases that depend on sibling fields.
    if template == "CrossSourceCompare":
        if field_name == "expected_entities":
            return FieldResult(
                field_name=field_name,
                status="INFO",
                expected=expected,
                actual=_l1_cross_source(bundle, "expected_entities"),
                detail="reference list (validated via expected_entities_min_matches)",
                layer="meta",
            )
        if field_name == "expected_entities_min_matches":
            actual_entities = _l1_cross_source(bundle, "expected_entities") or []
            exp_list = all_required.get("expected_entities", [])
            ok, detail = _match_min_matches(exp_list, actual_entities, int(expected))
            return FieldResult(
                field_name=field_name,
                status="PASS" if ok else "FAIL",
                expected=expected,
                actual=actual_entities,
                detail=detail,
                layer="L1",
            )

    # ------------------------------------------------------------------ L1
    extractor = _L1_DISPATCH.get(template)
    if extractor is None:
        return FieldResult(
            field_name=field_name,
            status="SKIP",
            expected=expected,
            actual=None,
            detail=f"unknown template {template!r}",
            layer="meta",
        )

    actual = extractor(bundle, field_name, answer)

    # ThreatContext's attack_techniques aggregation needs source docs.
    if (
        template == "ThreatContext"
        and field_name == "attack_techniques"
        and source_documents
    ):
        actual = _aggregate_attack_techniques(
            bundle.get("threat_context") or {}, source_documents
        )

    if actual == "__UNKNOWN__":
        return FieldResult(
            field_name=field_name,
            status="SKIP",
            expected=expected,
            actual=None,
            detail=f"field unmapped for template {template!r}",
            layer="meta",
        )

    # Pick the matcher by expected-value shape (spec §8.2).
    if field_name.endswith("_min_count"):
        # L1 min_count (e.g. associated_cves_min_count) — actual is int.
        ok, detail = _match_min_count(
            int(expected), actual if isinstance(actual, int) else 0
        )
    elif isinstance(expected, bool):
        ok, detail = _match_bool(expected, actual)
    elif isinstance(expected, (int, float)):
        ok, detail = _match_numeric(float(expected), actual)
    elif isinstance(expected, list):
        ok, detail = _match_list_any(expected, actual or [])
    elif isinstance(expected, str):
        ok, detail = _match_exact_ci(expected, actual)
    else:
        return FieldResult(
            field_name=field_name,
            status="SKIP",
            expected=expected,
            actual=actual,
            detail=f"unhandled expected type {type(expected).__name__}",
            layer="meta",
        )

    return FieldResult(
        field_name=field_name,
        status="PASS" if ok else "FAIL",
        expected=expected,
        actual=actual,
        detail=detail,
        layer="L1",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def evaluate_field_coverage(
    query_id: str,
    required_fields: dict,
    *,
    template: str | None,
    fact_bundle: dict | None,
    answer: str,
    source_documents: list[dict] | None = None,
) -> FieldCoverageResult:
    """Evaluate a single query's :class:`RAGResponse` against its required_fields.

    Parameters
    ----------
    query_id : str
        The eval-query ID (e.g. ``"vuln_004"``).
    required_fields : dict
        ``required_fields`` block from ``configs/eval_queries.yaml``.
    template : str | None
        Template name set by the router (``VulnTriage``, ``ThreatContext``,
        or ``CrossSourceCompare``). Required for L1 extraction dispatch.
    fact_bundle : dict | None
        Serialized Fact Bundle from ``RAGResponse.fact_bundle``.
    answer : str
        Final rendered answer (L1 block + L2 output concatenated), used
        for L2 min_count section parsing.
    source_documents : list[dict] | None
        Retrieved chunk dicts from ``RAGResponse.source_documents``. Only
        needed for ThreatContext's ``attack_techniques`` aggregation.
    """
    result = FieldCoverageResult(query_id=query_id, template=template)

    if not required_fields:
        return result

    if template is None:
        for field_name, expected in required_fields.items():
            result.results.append(
                FieldResult(
                    field_name=field_name,
                    status="SKIP",
                    expected=expected,
                    actual=None,
                    detail="no template assigned by router",
                    layer="meta",
                )
            )
        return result

    bundle = fact_bundle or {}

    for field_name, expected in required_fields.items():
        result.results.append(
            _evaluate_one_field(
                template=template,
                field_name=field_name,
                expected=expected,
                bundle=bundle,
                answer=answer,
                source_documents=source_documents,
                all_required=required_fields,
            )
        )

    return result


def aggregate_field_coverage(
    results: list[FieldCoverageResult],
) -> dict:
    """Build a summary table across per-query results (spec §8.2)."""
    per_template: dict[str, dict[str, int]] = {}
    per_layer: dict[str, dict[str, int]] = {}

    total_pass = total_fail = total_skip = total_info = 0
    ratios: list[float] = []

    for qr in results:
        tmpl = qr.template or "(unrouted)"
        slot = per_template.setdefault(tmpl, {"pass": 0, "fail": 0, "skip": 0})
        slot["pass"] += qr.pass_count
        slot["fail"] += qr.fail_count
        slot["skip"] += qr.skip_count

        for r in qr.results:
            bucket = per_layer.setdefault(
                r.layer, {"pass": 0, "fail": 0, "skip": 0}
            )
            if r.status == "PASS":
                bucket["pass"] += 1
            elif r.status == "FAIL":
                bucket["fail"] += 1
            elif r.status == "SKIP":
                bucket["skip"] += 1

        total_pass += qr.pass_count
        total_fail += qr.fail_count
        total_skip += qr.skip_count
        total_info += qr.info_count
        if qr.scored_count:
            ratios.append(qr.coverage_ratio)

    mean_coverage = sum(ratios) / len(ratios) if ratios else 0.0

    return {
        "queries_evaluated": len(results),
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_skip": total_skip,
        "total_info": total_info,
        "mean_coverage_ratio": round(mean_coverage, 4),
        "per_template": per_template,
        "per_layer": per_layer,
    }
