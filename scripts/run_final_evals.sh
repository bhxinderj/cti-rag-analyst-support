#!/usr/bin/env bash
# Final evaluation series (_final_v1) for the thesis Results chapter.
#
# Produces, on the post-ab48938 Setup-B index (84 991 chunks), with the
# frozen query set (ded35a8) and judge openrouter:openai/gpt-4o-mini:
#   - 3x RAGAS runs per pipeline row: hybrid templated, hybrid legacy,
#     bm25, vector, no-retrieval baseline  (report as mean +/- std;
#     single 28-query runs vary by ~0.05 from generation variance)
#   - Rubric rescoring of all six headline artifacts (axis 2)
#   - 3x field-coverage runs over the full query set (axis 3)
#
# Run from the repo root with the OpenRouter key in the environment:
#   caffeinate -is bash scripts/run_final_evals.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FAILURES=0

run() {
  echo "=== [$(date '+%F %T')] $*"
  if ! "$@"; then
    echo "=== BATCH-STEP-FAILED: $*"
    FAILURES=$((FAILURES + 1))
  fi
}

# --- Axis 1: RAGAS, headline pipelines first (paired per round) ---
for r in r1 r2 r3; do
  run env CTI_RAG_SETUP=b "$PY" main.py evaluate --mode hybrid --templated --name "final_v1_${r}"
  run env CTI_RAG_SETUP=b "$PY" main.py evaluate --mode hybrid --name "final_v1_${r}"
done

# --- Axis 1: ablation rows ---
for r in r1 r2 r3; do
  run env CTI_RAG_SETUP=b "$PY" main.py evaluate --mode bm25 --name "final_v1_${r}"
  run env CTI_RAG_SETUP=b "$PY" main.py evaluate --mode vector --name "final_v1_${r}"
  run env CTI_RAG_SETUP=b "$PY" main.py evaluate --mode hybrid --baseline --name "final_v1_${r}"
done

# --- Axis 2: rubric rescoring of the six headline artifacts ---
for f in data/evaluation_results/setup_b/ragas_hybrid_final_v1_r*.json \
         data/evaluation_results/setup_b/ragas_hybrid_templated_final_v1_r*.json; do
  [ -e "$f" ] || continue
  name="$(basename "$f" .json | sed -e 's/^ragas_//' -e 's/_[0-9]\{8\}_[0-9]\{6\}$//')"
  run env CTI_RAG_SETUP=b "$PY" main.py rubric --from-ragas-artifact "$f" --name "${name}"
done

# --- Axis 3: field coverage (exit 1 on individual L2 FAILs is expected) ---
for r in r1 r2 r3; do
  echo "=== [$(date '+%F %T')] field-coverage --all (${r})"
  env CTI_RAG_SETUP=b "$PY" tests/run_field_coverage_smoke.py --all --dump
  echo "=== field-coverage ${r} exit=$?"
done

echo "=== [$(date '+%F %T')] BATCH COMPLETE (failed steps: ${FAILURES})"
exit 0
