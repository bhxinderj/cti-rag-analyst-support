"""
Live Rubric LLM-as-Judge smoke runner (Phase-2, spec section 8.4).

Loads ``configs/eval_queries.yaml``, filters to the three calibrated
queries (``vuln_004``, ``ttp_001``, ``cross_003``), invokes
``RAGChain.query_templated`` for each, and scores the rendered answers
with :class:`RubricEvaluator`.

Unlike the unit tests in ``tests/test_rubric_eval.py`` (fake judge,
synthetic payloads), this runner exercises the full stack: real Ollama
generation, real Chroma retrieval, and the real configured judge
(OpenRouter / OpenAI / Ollama) per ``eval_llm_provider`` in
``configs/settings.yaml``.

Run
---
    CTI_RAG_SETUP=b .venv/bin/python tests/run_rubric_smoke.py
    CTI_RAG_SETUP=b .venv/bin/python tests/run_rubric_smoke.py --dump
    CTI_RAG_SETUP=b .venv/bin/python tests/run_rubric_smoke.py --only vuln_004

Outputs
-------
- Per-query table: dimension scores, total, justifications
- Cross-query aggregate: means per dimension + total
- With ``--dump``: ``data/interim/rubric_smoke_YYYYMMDDTHHMMSS.json`` for
  offline inspection alongside the field-coverage smoke output.

Exit code: non-zero if any sample errored (judge failure, parse failure,
or missing dimension).
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

from src.cti_rag.evaluation.rubric_eval import (  # noqa: E402
    DIMENSIONS,
    RubricEvaluator,
    RubricSample,
    aggregate_rubric_scores,
)
from src.cti_rag.rag.chain import RAGChain  # noqa: E402


CALIBRATION_IDS = ("vuln_004", "ttp_001", "cross_003")


def _print_divider(label: str) -> None:
    print(f"\n{'=' * 80}\n{label}\n{'=' * 80}")


def _load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("queries", [])


def _print_sample_report(result) -> None:
    header = f"{result.sample_id}  →  template={result.template}"
    _print_divider(header)
    if result.error:
        print(f"  ⚠️  error: {result.error}")
    total = result.scores.total()
    total_str = f"{total:.2f}" if total is not None else "  n/a"
    print(f"  total={total_str}")
    print()
    print(f"  {'Dimension':<18} {'Score':<6} Justification")
    print(f"  {'-' * 18} {'-' * 6} {'-' * 50}")
    for dim in DIMENSIONS:
        score = getattr(result.scores, dim)
        score_str = "-" if score is None else str(score)
        just = result.justifications.get(dim, "")
        if len(just) > 80:
            just = just[:77] + "..."
        print(f"  {dim:<18} {score_str:<6} {just}")


def _print_aggregate(summary: dict) -> None:
    _print_divider("AGGREGATE")
    overall = summary["overall"]
    print(f"  samples_scored       : {overall['n']}")
    print(f"  errors               : {overall['errors']}")
    for dim in DIMENSIONS:
        mean_val = overall.get(f"{dim}_mean")
        mean_str = f"{mean_val:.3f}" if isinstance(mean_val, (int, float)) else "n/a"
        print(f"  {dim+'_mean':<22} : {mean_str}")
    total_val = overall.get("total_mean")
    total_str = f"{total_val:.3f}" if isinstance(total_val, (int, float)) else "n/a"
    print(f"  total_mean           : {total_str}")

    if summary["per_template"]:
        print("\n  per_template:")
        for tmpl, stats in summary["per_template"].items():
            tot = stats.get("total_mean")
            tot_str = f"{tot:.3f}" if isinstance(tot, (int, float)) else "n/a"
            print(f"    {tmpl:<22} n={stats['n']}  total_mean={tot_str}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live rubric LLM-as-Judge smoke test")
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
        "--dump",
        action="store_true",
        help="Write the full per-sample result to data/interim/ as JSON.",
    )
    args = parser.parse_args()

    print(f"Active setup: {os.environ.get('CTI_RAG_SETUP', '(default)')}")
    print(f"Queries file: {args.queries_file}")

    all_queries = _load_queries(Path(args.queries_file))
    selected_ids = tuple(args.only) if args.only else CALIBRATION_IDS
    queries = [q for q in all_queries if q.get("id") in selected_ids]
    missing = [qid for qid in selected_ids if not any(q.get("id") == qid for q in queries)]
    if missing:
        print(f"⚠️  Missing query IDs in yaml: {missing}", file=sys.stderr)
    if not queries:
        print("No queries selected — nothing to evaluate.", file=sys.stderr)
        return 2

    chain = RAGChain(retrieval_mode="hybrid")
    evaluator = RubricEvaluator()
    print(f"Judge: {evaluator.judge_model_label}")

    results = []
    dump_records: list[dict] = []

    for query in queries:
        qid = query["id"]
        question = query["question"]
        template_expected = query.get("template_expected")
        ground_truth = query.get("ground_truth") or " ".join(
            point.strip() for point in (query.get("ground_truth_points") or []) if point.strip()
        )

        _print_divider(f"Running {qid}")
        print(f"  question          : {question}")
        print(f"  template_expected : {template_expected}")

        response = chain.query_templated(question)
        match_flag = "✓" if response.template == template_expected else "✗"
        print(
            f"  routed template   : {response.template}  "
            f"[{match_flag} expected={template_expected}]"
        )
        print(
            f"  timings           : retrieval={response.retrieval_time_ms:.0f}ms  "
            f"generation={response.generation_time_ms:.0f}ms  "
            f"total={response.total_time_ms:.0f}ms"
        )
        if response.abstention_reason:
            print(f"  ⚠️  abstained     : {response.abstention_reason}")

        sample = RubricSample(
            question=question,
            answer=response.answer,
            ground_truth=ground_truth,
            sample_id=qid,
            template=response.template,
            task_type=query.get("task_type"),
            difficulty=query.get("difficulty"),
        )
        result = evaluator.score_sample(sample)
        _print_sample_report(result)
        results.append(result)

        if args.dump:
            dump_records.append(
                {
                    "query": {
                        "id": qid,
                        "question": question,
                        "template_expected": template_expected,
                        "ground_truth": ground_truth,
                    },
                    "response": {
                        "template": response.template,
                        "routing_decision": response.routing_decision,
                        "abstention_reason": response.abstention_reason,
                        "retrieval_time_ms": response.retrieval_time_ms,
                        "generation_time_ms": response.generation_time_ms,
                        "total_time_ms": response.total_time_ms,
                        "l1_block": response.l1_block,
                        "l2_output": response.l2_output,
                        "answer": response.answer,
                    },
                    "rubric": result.to_dict(),
                }
            )

    summary = aggregate_rubric_scores(results)
    _print_aggregate(summary)

    if args.dump:
        out_dir = PROJECT_ROOT / "data" / "interim"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"rubric_smoke_{ts}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "judge_model": evaluator.judge_model_label,
                    "aggregate": summary,
                    "per_query": dump_records,
                },
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        print(f"\n📁 Dump written: {out_path.relative_to(PROJECT_ROOT)}")

    # Non-zero exit if any sample errored — useful for CI sanity.
    return 0 if summary["overall"]["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
