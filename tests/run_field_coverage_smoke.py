"""
Live Triage Field Coverage smoke runner (Phase-2, spec §8, §10 Step 10).

Loads ``configs/eval_queries.yaml``, filters to the three calibrated
queries (``vuln_004``, ``ttp_001``, ``cross_003``), invokes
``RAGChain.query_templated`` for each, and runs the response through
``evaluate_field_coverage`` / ``aggregate_field_coverage``.

Prints a per-query field breakdown plus the cross-query aggregate, and
— with ``--dump`` — serializes the full result to
``data/interim/field_coverage_smoke_YYYYMMDDTHHMMSS.json`` for offline
inspection.

Unlike the unit tests in ``tests/test_field_coverage.py`` (synthetic
bundles), this runner exercises the entire pipeline: real Ollama
generation, real Chroma retrieval, the router, entity-aware augmentation,
citation normalization, and the evaluator's L1/L2 extractors.

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/run_field_coverage_smoke.py
    CTI_RAG_SETUP=b .venv/bin/python tests/run_field_coverage_smoke.py --dump
    CTI_RAG_SETUP=b .venv/bin/python tests/run_field_coverage_smoke.py --only vuln_004
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CTI_RAG_SETUP", "b")

from src.cti_rag.evaluation.field_coverage import (  # noqa: E402
    aggregate_field_coverage,
    evaluate_field_coverage,
)
from src.cti_rag.rag.chain import RAGChain  # noqa: E402


CALIBRATION_IDS = ("vuln_004", "ttp_001", "cross_003")
_STATUS_GLYPH = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭ ", "INFO": "ℹ "}


def _print_divider(label: str) -> None:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")


def _load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("queries", [])


def _format_actual(actual) -> str:
    s = repr(actual)
    return s if len(s) <= 70 else s[:67] + "..."


def _print_query_report(qid: str, template: str | None, result) -> None:
    header = f"{qid}  →  template={template}"
    _print_divider(header)
    print(
        f"  pass={result.pass_count}  fail={result.fail_count}  "
        f"skip={result.skip_count}  info={result.info_count}  "
        f"coverage={result.coverage_ratio:.2%}"
    )
    print()
    print(f"  {'Field':<38} {'Layer':<6} {'Status':<7} Expected → Actual")
    print(f"  {'-' * 38} {'-' * 6} {'-' * 7} {'-' * 40}")
    for r in result.results:
        glyph = _STATUS_GLYPH.get(r.status, "?")
        exp = _format_actual(r.expected)
        act = _format_actual(r.actual)
        print(
            f"  {r.field_name:<38} {r.layer:<6} "
            f"{glyph} {r.status:<4} {exp} → {act}"
        )
        if r.status == "FAIL" and r.detail:
            print(f"    detail: {r.detail}")


def _print_aggregate(summary: dict) -> None:
    _print_divider("AGGREGATE")
    print(f"  queries_evaluated    : {summary['queries_evaluated']}")
    print(f"  total_pass           : {summary['total_pass']}")
    print(f"  total_fail           : {summary['total_fail']}")
    print(f"  total_skip           : {summary['total_skip']}")
    print(f"  total_info           : {summary['total_info']}")
    print(f"  mean_coverage_ratio  : {summary['mean_coverage_ratio']:.2%}")
    print("\n  per_template:")
    for tmpl, counts in summary["per_template"].items():
        print(f"    {tmpl:<22} pass={counts['pass']} fail={counts['fail']} skip={counts['skip']}")
    print("\n  per_layer:")
    for layer, counts in summary["per_layer"].items():
        print(f"    {layer:<22} pass={counts['pass']} fail={counts['fail']} skip={counts['skip']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live field-coverage smoke test")
    parser.add_argument(
        "--queries-file",
        default=str(PROJECT_ROOT / "configs" / "eval_queries.yaml"),
        help="Path to eval_queries.yaml",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Restrict to specific query IDs (repeatable). "
        "Default: all calibrated queries (vuln_004, ttp_001, cross_003).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every query in the file instead of the calibrated trio.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Write the full result to data/interim/ as JSON.",
    )
    args = parser.parse_args()

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}")
    print(f"Queries file: {args.queries_file}")

    all_queries = _load_queries(Path(args.queries_file))
    if args.all:
        queries = list(all_queries)
        selected_ids = tuple(q.get("id") for q in queries)
    else:
        selected_ids = tuple(args.only) if args.only else CALIBRATION_IDS
        queries = [q for q in all_queries if q.get("id") in selected_ids]
    missing = [qid for qid in selected_ids if not any(q.get("id") == qid for q in queries)]
    if missing:
        print(f"⚠️  Missing query IDs in yaml: {missing}", file=sys.stderr)
    if not queries:
        print("No queries selected — nothing to evaluate.", file=sys.stderr)
        return 2

    chain = RAGChain(retrieval_mode="hybrid")

    per_query_results = []
    dump_records: list[dict] = []

    for q in queries:
        qid = q["id"]
        question = q["question"]
        required_fields = q.get("required_fields") or {}
        template_expected = q.get("template_expected")

        _print_divider(f"Running {qid}")
        print(f"  question          : {question}")
        print(f"  template_expected : {template_expected}")
        print(f"  required_fields   : {list(required_fields.keys())}")

        resp = chain.query_templated(question)

        routed = resp.template
        match_flag = "✓" if routed == template_expected else "✗"
        print(f"  routed template   : {routed}  [{match_flag} expected={template_expected}]")
        print(f"  timings           : retrieval={resp.retrieval_time_ms:.0f}ms  "
              f"generation={resp.generation_time_ms:.0f}ms  total={resp.total_time_ms:.0f}ms")
        if resp.abstention_reason:
            print(f"  ⚠️  abstained     : {resp.abstention_reason}")

        fc = evaluate_field_coverage(
            qid,
            required_fields,
            template=resp.template,
            fact_bundle=resp.fact_bundle,
            answer=resp.answer,
            source_documents=resp.source_documents,
        )
        _print_query_report(qid, resp.template, fc)
        per_query_results.append(fc)

        if args.dump:
            dump_records.append(
                {
                    "query": {
                        "id": qid,
                        "question": question,
                        "template_expected": template_expected,
                        "required_fields": required_fields,
                    },
                    "response": {
                        "template": resp.template,
                        "routing_decision": resp.routing_decision,
                        "abstention_reason": resp.abstention_reason,
                        "retrieval_time_ms": resp.retrieval_time_ms,
                        "generation_time_ms": resp.generation_time_ms,
                        "total_time_ms": resp.total_time_ms,
                        "l1_block": resp.l1_block,
                        "l2_output": resp.l2_output,
                        "answer": resp.answer,
                        "fact_bundle": resp.fact_bundle,
                        "grounding_warnings": resp.grounding_warnings,
                    },
                    "field_coverage": fc.to_dict(),
                }
            )

    summary = aggregate_field_coverage(per_query_results)
    _print_aggregate(summary)

    if args.dump:
        out_dir = PROJECT_ROOT / "data" / "interim"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"field_coverage_smoke_{ts}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {"summary": summary, "per_query": dump_records},
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"\n📁 Dump written: {out_path.relative_to(PROJECT_ROOT)}")

    # Exit code: non-zero if any scored field failed — useful for CI.
    return 0 if summary["total_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
