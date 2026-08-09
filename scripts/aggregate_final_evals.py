#!/usr/bin/env python3
"""
Aggregate the _final_v1 evaluation artifacts into thesis-ready statistics.

Complements scripts/eval_to_tex_table.py (which renders display tables)
with the numbers the Results/Discussion text needs beyond mean +/- std:

  RAGAS rows:
    - per-metric mean/std over runs, NaN-aware
    - effective n per metric (non-NaN samples out of 28, averaged over runs)
    - answer_relevancy zero counts per run (noncommittal-classifier hits)
    - abstention counts per run
  Rubric rows:
    - dimension means/stds over runs
    - per-template breakdown via offline join with template_expected from
      configs/eval_queries.yaml (the offline rescoring path loses the
      routed template, so per_template in the artifacts is empty)

Writes data/interim/final_v1_aggregate.json and prints a readable report.

Usage:
    .venv/bin/python scripts/aggregate_final_evals.py
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "data" / "evaluation_results" / "setup_b"
QUERY_FILE = REPO_ROOT / "configs" / "eval_queries.yaml"
OUT_FILE = REPO_ROOT / "data" / "interim" / "final_v1_aggregate.json"

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]

RAGAS_GROUPS = {
    "templated_hybrid": "ragas_hybrid_templated_final_v1_r*.json",
    "legacy_hybrid": "ragas_hybrid_final_v1_r*.json",
    "bm25_only": "ragas_bm25_final_v1_r*.json",
    "vector_only": "ragas_vector_final_v1_r*.json",
    "baseline": "ragas_baseline_final_v1_r*.json",
    "templated_hostedgen_haiku45": "ragas_hybrid_templated_hostedgen_haiku45_final_v1*.json",
}

RUBRIC_GROUPS = {
    "templated_hybrid": "rubric_hybrid_templated_final_v1_r*.json",
    "legacy_hybrid": "rubric_hybrid_final_v1_r*.json",
    "baseline": "rubric_baseline_final_v1_r*.json",
    "templated_hostedgen_haiku45": "rubric_hybrid_templated_hostedgen_haiku45_final_v1_*.json",
}

RUBRIC_DIMS = ["prioritization", "actionability", "completeness", "traceability"]

ZERO_THRESHOLD = 0.05


def _load(pattern: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(str(ARTIFACT_DIR / pattern)))]


def _is_num(value) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": round(st.mean(values), 4),
        "std": round(st.stdev(values), 4) if len(values) >= 2 else None,
    }


def aggregate_ragas() -> dict:
    result = {}
    for group, pattern in RAGAS_GROUPS.items():
        runs = _load(pattern)
        if not runs:
            continue
        entry: dict = {"n_runs": len(runs), "metrics": {}}

        for metric in RAGAS_METRICS:
            run_means = [r["metrics"][metric] for r in runs if metric in r["metrics"]]
            if not run_means:
                continue
            valid_counts = []
            for r in runs:
                samples = [
                    s["ragas_metrics"].get(metric)
                    for s in r["per_sample"]
                    if s.get("ragas_metrics")
                ]
                valid_counts.append(sum(1 for v in samples if _is_num(v)))
            entry["metrics"][metric] = {
                **_mean_std(run_means),
                "per_run": [round(v, 4) for v in run_means],
                "effective_n": round(st.mean(valid_counts), 1) if valid_counts else None,
            }

        entry["relevancy_zero_counts"] = [
            sum(
                1
                for s in r["per_sample"]
                if _is_num(s.get("ragas_metrics", {}).get("answer_relevancy"))
                and s["ragas_metrics"]["answer_relevancy"] < ZERO_THRESHOLD
            )
            for r in runs
        ]
        entry["abstention_counts"] = [
            sum(1 for s in r["per_sample"] if s.get("abstention_reason")) for r in runs
        ]
        result[group] = entry
    return result


def aggregate_rubric() -> dict:
    queries = yaml.safe_load(open(QUERY_FILE))["queries"]
    template_by_id = {q["id"]: q.get("template_expected") for q in queries}

    result = {}
    for group, pattern in RUBRIC_GROUPS.items():
        runs = _load(pattern)
        if not runs:
            continue
        entry: dict = {"n_runs": len(runs), "dimensions": {}, "per_template": {}}

        for dim in RUBRIC_DIMS + ["total"]:
            key = f"{dim}_mean" if dim != "total" else "total_mean"
            vals = [r["aggregated"][key] for r in runs]
            entry["dimensions"][dim] = {**_mean_std(vals), "per_run": [round(v, 4) for v in vals]}

        # Offline join: the rescoring path loses the routed template, so we
        # group per-sample totals by template_expected from the query set.
        totals_by_template: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            for s in r["per_sample"]:
                if s.get("error"):
                    continue
                template = s.get("template") or template_by_id.get(s.get("sample_id"))
                if template and _is_num(s.get("total")):
                    totals_by_template[template].append(s["total"])
        entry["per_template"] = {
            template: {**_mean_std(vals), "n_samples": len(vals)}
            for template, vals in sorted(totals_by_template.items())
        }
        result[group] = entry
    return result


def main() -> int:
    aggregate = {"ragas": aggregate_ragas(), "rubric": aggregate_rubric()}

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_FILE}\n")

    for group, entry in aggregate["ragas"].items():
        zeros = entry["relevancy_zero_counts"]
        absts = entry["abstention_counts"]
        print(f"[RAGAS] {group} (n={entry['n_runs']}) zeros/run={zeros} abstentions/run={absts}")
        for metric, stats in entry["metrics"].items():
            std = f" ± {stats['std']:.3f}" if stats["std"] is not None else ""
            print(
                f"    {metric:<20} {stats['mean']:.3f}{std}   eff_n={stats['effective_n']}"
            )
    print()
    for group, entry in aggregate["rubric"].items():
        print(f"[Rubric] {group} (n={entry['n_runs']})")
        for template, stats in entry["per_template"].items():
            std = f" ± {stats['std']:.2f}" if stats["std"] is not None else ""
            print(
                f"    {template:<20} total {stats['mean']:.2f}{std}  (n_samples={stats['n_samples']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
