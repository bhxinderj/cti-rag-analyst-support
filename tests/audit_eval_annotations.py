"""
Diagnostic audit: cross-check ``configs/eval_queries.yaml`` annotations
against the persisted Setup-B index.

Motivation
----------
The L1 smoke test surfaced that ``vuln_004.required_fields.cvss_score``
claims 9.4 while NVD actually indexes 7.5 for CVE-2023-4966. That
mismatch would produce spurious FAILs downstream in the Field-Coverage
Evaluator (Sprint 3). Better to find every such discrepancy now — before
we build the evaluator on a shaky annotation layer.

What this does
--------------
For every query in the eval set, it:
  1. Extracts CVE IDs mentioned in the question and ground_truth.
  2. Pulls each CVE's NVD + KEV metadata from the index.
  3. For queries with ``required_fields``, compares each annotation
     against the index-truth value.
  4. For queries without ``required_fields``, scans the ``ground_truth``
     for literal "CVSS X.Y" claims and compares them against the index.

Purely a reporting tool. No side effects.

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/audit_eval_annotations.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CTI_RAG_SETUP", "b")

from src.cti_rag.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", flags=re.IGNORECASE)
_CVSS_CLAIM_RE = re.compile(r"CVSS\s*([0-9]+\.[0-9])", flags=re.IGNORECASE)


def _fetch_index_truth(collection, cve: str) -> dict[str, Any]:
    """Return the aggregated NVD + KEV truth for a single CVE."""
    res = collection.get(where={"cve_ids": {"$eq": cve}}, include=["metadatas"])
    nvd_meta = None
    kev_meta = None
    advisory_count = 0
    misp_count = 0
    for meta in res.get("metadatas") or []:
        src = meta.get("source", "")
        if src == "nvd" and nvd_meta is None:
            nvd_meta = meta
        elif src == "cisa_kev" and kev_meta is None:
            kev_meta = meta
        elif src == "cisa_advisory":
            advisory_count += 1
        elif src == "misp":
            misp_count += 1
    return {
        "cve_id": cve,
        "in_nvd": nvd_meta is not None,
        "in_kev": kev_meta is not None,
        "cvss_score": nvd_meta.get("cvss_score") if nvd_meta else None,
        "cwe_ids": nvd_meta.get("cwe_ids") if nvd_meta else None,
        "nvd_severity": nvd_meta.get("severity") if nvd_meta else None,
        "nvd_affected_products": nvd_meta.get("affected_products") if nvd_meta else None,
        "kev_ransomware": kev_meta.get("known_ransomware") if kev_meta else None,
        "kev_ransomware_bool": (
            bool(kev_meta.get("known_ransomware_bool")) if kev_meta else None
        ),
        "advisory_chunks": advisory_count,
        "misp_chunks": misp_count,
    }


def _audit_required_fields(qid: str, required: dict, truth: dict) -> list[str]:
    """Return a list of human-readable mismatch descriptions."""
    issues: list[str] = []

    if "cvss_score" in required:
        expected = required["cvss_score"]
        actual = truth["cvss_score"]
        if actual is None:
            issues.append(f"cvss_score: annotation={expected} but CVE not in NVD index")
        elif abs(float(actual) - float(expected)) > 0.1:
            issues.append(
                f"cvss_score: annotation={expected} but NVD index={actual} "
                f"(Δ={float(actual) - float(expected):+.1f})"
            )

    if "cwe_ids" in required:
        expected_cwes = set(str(c).upper() for c in required["cwe_ids"])
        raw = truth["cwe_ids"] or ""
        actual_cwes = set(c.strip().upper() for c in str(raw).split(",") if c.strip())
        missing = expected_cwes - actual_cwes
        if missing:
            issues.append(
                f"cwe_ids: annotation wants {sorted(expected_cwes)} but "
                f"NVD has {sorted(actual_cwes)} (missing {sorted(missing)})"
            )

    if "kev_listed" in required:
        if bool(required["kev_listed"]) != bool(truth["in_kev"]):
            issues.append(
                f"kev_listed: annotation={required['kev_listed']} but "
                f"index in_kev={truth['in_kev']}"
            )

    if "ransomware_use" in required:
        if bool(required["ransomware_use"]) != bool(truth["kev_ransomware_bool"]):
            issues.append(
                f"ransomware_use: annotation={required['ransomware_use']} but "
                f"KEV bool={truth['kev_ransomware_bool']} "
                f"(raw={truth['kev_ransomware']!r})"
            )

    if "triage_signal" in required:
        # Recompute the deterministic severity signal from truth.
        from src.cti_rag.rag.severity import compute_triage_signal

        sig = compute_triage_signal(
            cvss_score=truth["cvss_score"],
            kev_listed=bool(truth["in_kev"]),
            ransomware_use=bool(truth["kev_ransomware_bool"]),
        )
        if sig.severity != required["triage_signal"]:
            issues.append(
                f"triage_signal: annotation={required['triage_signal']!r} but "
                f"deterministic rule says {sig.severity!r} "
                f"(CVSS={truth['cvss_score']}, KEV={truth['in_kev']}, "
                f"ransomware={truth['kev_ransomware_bool']}; rationale: {sig.rationale})"
            )

    return issues


def _scan_ground_truth_cvss_claims(
    gt: str, cves: list[str], truths: dict[str, dict]
) -> list[str]:
    """Light check: any literal 'CVSS X.Y' tokens that clash with index."""
    issues: list[str] = []
    if not gt:
        return issues
    # This is intentionally naive: we just check that every CVSS claim
    # appears somewhere in the truth set. Deeper per-CVE attribution is
    # not attempted (the YAML prose doesn't always place each claim next
    # to its CVE).
    claimed_scores = {float(m.group(1)) for m in _CVSS_CLAIM_RE.finditer(gt)}
    if not claimed_scores:
        return issues
    actual_scores = {
        float(t["cvss_score"])
        for t in truths.values()
        if t["cvss_score"] is not None
    }
    orphans = {s for s in claimed_scores if s not in actual_scores}
    if orphans:
        issues.append(
            f"ground_truth cites CVSS {sorted(orphans)} but index has "
            f"{sorted(actual_scores) or 'no NVD entry'} for {cves}"
        )
    return issues


def main() -> int:
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "eval_queries.yaml").read_text())
    queries = data["queries"]

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}")
    print(f"Queries to audit: {len(queries)}\n")

    retriever = HybridRetriever(retrieval_mode="hybrid")
    collection = retriever.collection

    # Pre-cache all CVE lookups to avoid repeat index hits.
    all_cves: set[str] = set()
    per_query_cves: dict[str, list[str]] = {}
    for q in queries:
        text = f"{q.get('question', '')} {q.get('ground_truth', '')}"
        cves = sorted(set(m.group(0).upper() for m in _CVE_RE.finditer(text)))
        per_query_cves[q["id"]] = cves
        all_cves.update(cves)

    print(f"Unique CVEs referenced across all queries: {len(all_cves)}")
    truths: dict[str, dict] = {}
    for cve in sorted(all_cves):
        truths[cve] = _fetch_index_truth(collection, cve)

    missing_from_index = [c for c, t in truths.items() if not t["in_nvd"] and not t["in_kev"]]
    if missing_from_index:
        print(f"⚠️  CVEs not in index at all: {missing_from_index}")
    print()

    total_issues = 0
    queries_with_issues = 0

    for q in queries:
        qid = q["id"]
        cves = per_query_cves[qid]
        local_truths = {c: truths[c] for c in cves}

        issues: list[str] = []

        req = q.get("required_fields")
        if req and cves:
            # required_fields queries always target a single CVE.
            primary_cve = cves[0]
            issues.extend(_audit_required_fields(qid, req, truths[primary_cve]))
        # Note: queries like ttp_001 annotate required_fields on a
        # tactic/technique, not a CVE — no CVE-truth audit applies there.

        gt_issues = _scan_ground_truth_cvss_claims(
            q.get("ground_truth", ""), cves, local_truths
        )
        issues.extend(gt_issues)

        if issues:
            queries_with_issues += 1
            total_issues += len(issues)
            print(f"━━━ {qid} ({q.get('task_type')}) ━━━")
            for i in issues:
                print(f"  ✗ {i}")
            if cves:
                for c in cves:
                    t = truths[c]
                    print(
                        f"    · index truth for {c}: "
                        f"in_nvd={t['in_nvd']}, cvss={t['cvss_score']}, "
                        f"cwe={t['cwe_ids']!r}, in_kev={t['in_kev']}, "
                        f"ransom_bool={t['kev_ransomware_bool']}"
                    )
            print()

    print("═" * 80)
    print(
        f"Audit complete: {queries_with_issues}/{len(queries)} queries have "
        f"annotation mismatches, {total_issues} total issues."
    )
    print("═" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
