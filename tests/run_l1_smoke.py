"""
Layer-1 smoke test for the three Phase-2 calibration queries.

Goal
----
Before wiring the new templates into ``chain.py`` and burning LLM / RAGAS
budget, verify that the **deterministic** half of the pipeline is
already working end-to-end against the real Setup-B index:

    question  ─► router  ─► hybrid retriever  ─► fact bundle  ─► L1 block
                                                       │
                                                       ▼
                                        required_fields validation
                                        (from configs/eval_queries.yaml)

If every calibration case's Layer-1 fields satisfy its
``required_fields`` annotation, the Fact Bundle extraction + Severity
Signal logic are production-ready and we can proceed to Sprint 3 with
confidence. Any failure here points at a metadata or aggregation gap
that would invalidate the field-coverage evaluation downstream.

This script deliberately does NOT invoke the LLM. It measures only L1
coverage; ``_min_count`` fields that refer to LLM-generated sections
(e.g. ``mitigations_min_count``) are reported as "L2 — skipped".

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/run_l1_smoke.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Setup B is active unless the caller picked something else.
os.environ.setdefault("CTI_RAG_SETUP", "b")

from src.cti_rag.rag.facts import (  # noqa: E402
    build_fact_bundle,
    extract_attack_techniques,
    extract_cve_ids,
)
from src.cti_rag.rag.router import classify_query  # noqa: E402
from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


# Smoke test bumps the rerank top-k so the L1 layer gets a representative
# slice of sources (NVD + KEV + MISP + Advisory). Production chain.py keeps
# the config-driven top_k; retrieval recall at top-5 is a separate gap
# tracked for Sprint 3 (source-diverse retrieval / entity-aware boost).
_SMOKE_RERANK_TOP_K = 20
_SMOKE_FUSION_TOP_K = 40


CALIBRATION_IDS = {"vuln_004", "ttp_001", "cross_003"}

# Fields whose validation requires inspecting LLM output, not the Fact
# Bundle. Treated as "skipped" by this L1-only smoke test.
_L2_ONLY_SUFFIXES = (
    "_min_count",  # e.g. mitigations_min_count, detection_or_mitigation_min_count,
    #         cross_source_facts_min_count — all refer to rendered L2 output
)
_L2_ONLY_EXCEPTIONS = {
    # _min_count fields that ARE deterministic (read from Fact Bundle)
    "associated_cves_min_count",  # ThreatContext L1 aggregation
    "expected_entities_min_matches",  # CrossSourceCompare L1 coverage
}


# ---------------------------------------------------------------------------
# Matching helpers (spec §8.2 "Variante Y")
# ---------------------------------------------------------------------------


def _match_numeric(expected: float, actual) -> tuple[bool, str]:
    if actual is None:
        return False, "actual is None"
    try:
        return (
            abs(float(actual) - float(expected)) <= 0.1,
            f"actual={actual}, expected={expected} (tol 0.1)",
        )
    except (TypeError, ValueError):
        return False, f"actual={actual!r} not numeric"


def _match_bool(expected: bool, actual) -> tuple[bool, str]:
    return bool(actual) == bool(expected), f"actual={actual}, expected={expected}"


def _match_exact_ci(expected: str, actual) -> tuple[bool, str]:
    if actual is None:
        return False, "actual is None"
    return (
        str(actual).strip().lower() == str(expected).strip().lower(),
        f"actual={actual!r}, expected={expected!r}",
    )


def _match_list_any(expected: list, actual) -> tuple[bool, str]:
    """any_substring: at least one expected item appears in actual list."""
    if not actual:
        return False, "actual is empty"
    actual_norm = [str(x).strip().lower() for x in actual]
    for exp in expected:
        exp_norm = str(exp).strip().lower()
        for actual_item in actual_norm:
            if exp_norm == actual_item or exp_norm in actual_item or actual_item in exp_norm:
                return True, f"matched {exp!r} in {actual}"
    return False, f"none of {expected} in {actual}"


def _match_min_count(expected: int, actual_count: int) -> tuple[bool, str]:
    return actual_count >= int(expected), f"count={actual_count}, required ≥ {expected}"


def _match_min_matches(expected_list: list, actual_list: list, required: int) -> tuple[bool, str]:
    actual_norm = {str(x).strip().upper() for x in actual_list}
    hits = [e for e in expected_list if str(e).strip().upper() in actual_norm]
    return (
        len(hits) >= int(required),
        f"matched {len(hits)}/{len(expected_list)}, required ≥ {required}: {hits}",
    )


# ---------------------------------------------------------------------------
# Per-template field extraction
# ---------------------------------------------------------------------------


def _bundle_vuln_triage_field(bundle, field: str):
    if not bundle.vuln_triage:
        return None
    primary = bundle.vuln_triage[0]
    mapping = {
        "cve_id": primary.cve_id,
        "cvss_score": primary.cvss_score,
        "cwe_ids": primary.cwe_ids,
        "affected_products": primary.affected_products,
        "kev_listed": primary.kev_listed,
        "ransomware_use": primary.ransomware_use,
        "triage_signal": (primary.triage_signal or {}).get("severity"),
    }
    return mapping.get(field, "__UNKNOWN__")


def _bundle_threat_context_field(bundle, field: str, chunks: list[dict] | None = None):
    tc = bundle.threat_context
    if tc is None:
        return None
    # For tactic-kind queries the primary entity is a TA-id; associated
    # techniques must be aggregated from the retrieved chunks' metadata
    # and content. tc.attack_technique is only populated for technique-kind.
    aggregated_techniques: list[str] = []
    if tc.attack_technique:
        aggregated_techniques.append(tc.attack_technique)
    if chunks:
        for c in chunks:
            meta = c.get("metadata") or {}
            raw = meta.get("attack_techniques") or ""
            if isinstance(raw, str):
                for tid in raw.split(","):
                    tid = tid.strip().upper()
                    if tid and tid not in aggregated_techniques:
                        aggregated_techniques.append(tid)
            # Also fall back to content regex for advisories that list
            # techniques inline without metadata normalisation.
            for tid in extract_attack_techniques(
                (c.get("title") or "") + " " + (c.get("content") or "")
            ):
                if tid not in aggregated_techniques:
                    aggregated_techniques.append(tid)
    mapping = {
        "attack_techniques": aggregated_techniques,
        "attack_tactic": (
            [tc.tactic_id, tc.tactic_name] if (tc.tactic_id or tc.tactic_name) else []
        ),
        "associated_cves_min_count": len(tc.associated_cves),
    }
    return mapping.get(field, "__UNKNOWN__")


def _bundle_cross_source_field(bundle, field: str):
    cs = bundle.cross_source
    if cs is None:
        return None
    mapping = {
        "expected_entities": [e.cve_id for e in cs.entities],
    }
    return mapping.get(field, "__UNKNOWN__")


# ---------------------------------------------------------------------------
# Validation dispatch
# ---------------------------------------------------------------------------


def _validate_required_fields(
    query_id: str, template: str, bundle, required: dict, chunks: list[dict] | None = None
):
    results: list[tuple[str, str, str]] = []  # (field, status, detail)

    for field, expected in required.items():
        # Defer LLM-output-dependent checks.
        if any(field.endswith(suf) for suf in _L2_ONLY_SUFFIXES) and field not in _L2_ONLY_EXCEPTIONS:
            results.append((field, "SKIP (L2)", f"requires LLM output: expected ≥ {expected}"))
            continue

        if template == "VulnTriage":
            actual = _bundle_vuln_triage_field(bundle, field)
        elif template == "ThreatContext":
            actual = _bundle_threat_context_field(bundle, field, chunks)
        elif template == "CrossSourceCompare":
            if field == "expected_entities":
                # Stored as reference list; validated via _min_matches below.
                results.append((field, "INFO", f"reference list: {expected}"))
                continue
            if field == "expected_entities_min_matches":
                actual_entities = _bundle_cross_source_field(bundle, "expected_entities") or []
                exp_list = required.get("expected_entities", [])
                ok, detail = _match_min_matches(exp_list, actual_entities, expected)
                results.append((field, "PASS" if ok else "FAIL", detail))
                continue
            actual = _bundle_cross_source_field(bundle, field)
        else:
            results.append((field, "SKIP", f"unknown template {template}"))
            continue

        if actual == "__UNKNOWN__":
            results.append((field, "SKIP", f"field unmapped for {template}"))
            continue

        # Pick the matcher by expected value shape.
        if field.endswith("_min_count"):
            ok, detail = _match_min_count(expected, actual if isinstance(actual, int) else 0)
        elif isinstance(expected, bool):
            ok, detail = _match_bool(expected, actual)
        elif isinstance(expected, (int, float)):
            ok, detail = _match_numeric(float(expected), actual)
        elif isinstance(expected, list):
            ok, detail = _match_list_any(expected, actual or [])
        elif isinstance(expected, str):
            ok, detail = _match_exact_ci(expected, actual)
        else:
            ok, detail = False, f"unhandled expected type {type(expected).__name__}"

        results.append((field, "PASS" if ok else "FAIL", detail))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _auto_detect_cves_from_chunks(chunks: list[dict], top_n: int = 10) -> list[str]:
    """For CrossSourceCompare queries without explicit CVE IDs."""
    counts: Counter[str] = Counter()
    for chunk in chunks:
        text = (chunk.get("title") or "") + " " + (chunk.get("content") or "")
        for cve in extract_cve_ids(text):
            counts[cve] += 1
        # Also consider metadata.cve_ids for robustness.
        meta = chunk.get("metadata") or {}
        raw = meta.get("cve_ids")
        if isinstance(raw, str):
            for cve in raw.split(","):
                cve = cve.strip().upper()
                if cve:
                    counts[cve] += 1
    return [cve for cve, _ in counts.most_common(top_n)]


def _fetch_chunks_by_cves(retriever, cves: list[str]) -> list[dict]:
    """
    Metadata-first fallback for CVE-centric queries. Guarantees the primary
    NVD / KEV chunk for each CVE is present in the L1 bundle, independent
    of hybrid-ranker recall. This is a smoke-test scaffold — production
    chain.py must resolve recall via entity-aware re-ranking (Sprint 3),
    not by bypassing the retriever.
    """
    fetched: list[dict] = []
    seen_ids: set[str] = set()
    for cve in cves:
        res = retriever.collection.get(where={"cve_ids": {"$eq": cve}}, limit=20)
        for doc_id, content, meta in zip(
            res.get("ids", []) or [],
            res.get("documents", []) or [],
            res.get("metadatas", []) or [],
        ):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            meta = dict(meta)
            fetched.append(
                {
                    "doc_id": doc_id,
                    "content": content or "",
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title", ""),
                    "citation_label": f"{meta.get('title', '')} | {doc_id}".strip(" |"),
                    "metadata": meta,
                }
            )
    return fetched


def _chunk_to_dict(chunk) -> dict:
    meta = dict(chunk.metadata)
    return {
        "doc_id": chunk.doc_id,
        "content": chunk.content,
        "source": meta.get("source", "unknown"),
        "title": meta.get("title", ""),
        "citation_label": f"{meta.get('title', '')} | {chunk.doc_id}".strip(" |"),
        "metadata": meta,
    }


def main() -> int:
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_queries.yaml").read_text())
    queries = [q for q in data["queries"] if q["id"] in CALIBRATION_IDS]
    queries.sort(key=lambda q: ["vuln_004", "ttp_001", "cross_003"].index(q["id"]))

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}\n")
    retriever = HybridRetriever(retrieval_mode="hybrid")
    # Give L1 enough source diversity; retrieval recall at default top-5 is
    # a separate, acknowledged Sprint-3 concern (source-diverse re-ranking).
    retriever.fusion_top_k = max(retriever.fusion_top_k, _SMOKE_FUSION_TOP_K)
    retriever.rerank_top_k = max(retriever.rerank_top_k, _SMOKE_RERANK_TOP_K)
    print(
        f"Smoke-test overrides: fusion_top_k={retriever.fusion_top_k}, "
        f"rerank_top_k={retriever.rerank_top_k}\n"
    )

    overall_pass = 0
    overall_fail = 0
    overall_skip = 0

    for q in queries:
        print("=" * 80)
        print(f"Query [{q['id']}]: {q['question']}")
        decision = classify_query(q["question"])
        print(
            f"Router → template={decision.template} "
            f"rule={decision.rule_matched} "
            f"primary_entities={decision.primary_entities} "
            f"kind={decision.threat_context_kind}"
        )

        raw_chunks = retriever.retrieve(q["question"])
        chunks = [_chunk_to_dict(c) for c in raw_chunks]
        print(f"Retrieved: {len(chunks)} chunks")
        for c in chunks[:5]:
            print(f"  - {c['doc_id']} ({c['source']}): {c['title']}")
        if len(chunks) > 5:
            print(f"  ... +{len(chunks) - 5} more")

        primary_entities = decision.primary_entities
        if decision.template == "CrossSourceCompare" and not primary_entities:
            primary_entities = _auto_detect_cves_from_chunks(chunks)
            print(f"Auto-detected entities for CrossSourceCompare: {primary_entities}")

        # Entity-aware recall scaffolding: for CVE-centric templates, ensure
        # the primary NVD / KEV chunks for each target CVE are present so
        # L1 has a fair chance. Sprint-3 retrieval work will move this into
        # the ranker; here it simply isolates L1 under test.
        if decision.template in ("VulnTriage", "CrossSourceCompare") and primary_entities:
            cve_entities = [e for e in primary_entities if e.upper().startswith("CVE-")]
            if cve_entities:
                extra = _fetch_chunks_by_cves(retriever, cve_entities)
                existing_ids = {c["doc_id"] for c in chunks}
                added = [c for c in extra if c["doc_id"] not in existing_ids]
                if added:
                    print(f"Added {len(added)} entity-filtered chunks for {cve_entities}")
                    chunks = chunks + added

        bundle = build_fact_bundle(
            decision.template,
            primary_entities,
            chunks,
            threat_context_kind=decision.threat_context_kind,
        )

        # Annotate severity signals on any VulnTriage entities.
        if decision.template == "VulnTriage":
            from src.cti_rag.rag.severity import annotate_vuln_triage_bundle

            for facts in bundle.vuln_triage:
                annotate_vuln_triage_bundle(facts)
        elif decision.template == "CrossSourceCompare" and bundle.cross_source:
            from src.cti_rag.rag.severity import annotate_vuln_triage_bundle

            for facts in bundle.cross_source.entities:
                annotate_vuln_triage_bundle(facts)

        required = q.get("required_fields") or {}
        if not required:
            print("⚠️  No required_fields annotation — skipping validation")
            continue

        results = _validate_required_fields(
            q["id"], decision.template, bundle, required, chunks=chunks
        )

        print("\nRequired field coverage (L1 only):")
        for field, status, detail in results:
            marker = {"PASS": "✓", "FAIL": "✗", "SKIP (L2)": "—", "SKIP": "—", "INFO": "·"}.get(
                status, "?"
            )
            print(f"  {marker} [{status:10}] {field:35} {detail}")

            if status == "PASS":
                overall_pass += 1
            elif status == "FAIL":
                overall_fail += 1
            elif status.startswith("SKIP"):
                overall_skip += 1

        print()

    print("=" * 80)
    print(
        f"Overall L1 field coverage: {overall_pass} pass, "
        f"{overall_fail} fail, {overall_skip} skipped (L2 or unmapped)"
    )
    print("=" * 80)
    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
