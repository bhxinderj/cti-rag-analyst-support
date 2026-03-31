"""Run a live end-to-end smoke check against local Ollama and real artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cti_rag.evaluation.ragas_eval import (
    RAGASEvaluator,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.cti_rag.rag.chain import RAGChain
from src.cti_rag.utils.config import load_config, require_local_hf_snapshot


SMOKE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "answer_correctness": answer_correctness,
    "context_precision": context_precision,
    "context_recall": context_recall,
}


def _preflight_ollama(config: dict, *, require_eval_model: bool) -> None:
    base_url = config["llm"]["base_url"].rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. Start `ollama serve` before running the live smoke check."
        ) from exc

    payload = response.json()
    model_names = {model.get("name", "") for model in payload.get("models", [])}

    llm_model = config["llm"]["model_name"]
    if llm_model not in model_names:
        raise RuntimeError(f"Ollama model '{llm_model}' is not available locally.")

    if require_eval_model:
        eval_model = config["evaluation"]["ragas"].get("eval_llm", llm_model)
        if eval_model not in model_names:
            raise RuntimeError(f"Ollama evaluation model '{eval_model}' is not available locally.")


def _preflight_retrieval_artifacts(config: dict) -> None:
    root = PROJECT_ROOT

    require_local_hf_snapshot(config["embedding"]["model_name"], artifact_label="Embedding model")
    require_local_hf_snapshot(config["retrieval"]["reranker"]["model_name"], artifact_label="Reranker model")

    bm25_path = root / config["retrieval"]["bm25"]["index_path"]
    chroma_path = root / config["chromadb"]["persist_directory"]
    if not bm25_path.exists():
        raise RuntimeError(f"BM25 artifact is missing: {bm25_path}")
    if not chroma_path.exists():
        raise RuntimeError(f"ChromaDB persist directory is missing: {chroma_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live CTI-RAG smoke check")
    parser.add_argument("--setup", choices=["default", "a", "b"], default="b")
    parser.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    parser.add_argument("--question", default="What is CVE-2024-3094?")
    parser.add_argument("--with-eval", action="store_true", help="Also run a one-sample RAGAS evaluation.")
    parser.add_argument(
        "--eval-metrics",
        default="answer_relevancy",
        help=(
            "Comma-separated RAGAS metrics for the live smoke evaluation. "
            "Defaults to answer_relevancy for a faster, more stable smoke run."
        ),
    )
    args = parser.parse_args()

    if args.setup == "default":
        os.environ.pop("CTI_RAG_SETUP", None)
    else:
        os.environ["CTI_RAG_SETUP"] = args.setup

    config = load_config()
    snapshot_date = config["data"].get("snapshot_date")

    _preflight_ollama(config, require_eval_model=args.with_eval)
    _preflight_retrieval_artifacts(config)

    print(f"Running live smoke check for setup={args.setup} mode={args.mode}")

    baseline_chain = RAGChain(retrieval_mode=args.mode)
    baseline = baseline_chain.query_baseline(args.question, snapshot_date=snapshot_date)
    print(json.dumps({
        "baseline_generation_time_ms": round(baseline.generation_time_ms, 2),
        "baseline_answer_preview": baseline.answer[:160],
    }, indent=2))

    rag_chain = RAGChain(retrieval_mode=args.mode)
    rag = rag_chain.query(args.question)
    print(json.dumps({
        "retrieval_mode": rag.retrieval_mode,
        "retrieval_time_ms": round(rag.retrieval_time_ms, 2),
        "generation_time_ms": round(rag.generation_time_ms, 2),
        "total_time_ms": round(rag.total_time_ms, 2),
        "num_contexts": len(rag.contexts),
        "num_cited_sources": sum(1 for doc in rag.source_documents if doc.get("cited_in_answer")),
        "abstention_reason": rag.abstention_reason,
        "answer_preview": rag.answer[:160],
    }, indent=2))

    if args.with_eval:
        metric_names = [name.strip() for name in args.eval_metrics.split(",") if name.strip()]
        unknown_metrics = [name for name in metric_names if name not in SMOKE_METRICS]
        if unknown_metrics:
            raise ValueError(
                f"Unsupported eval metric(s): {', '.join(unknown_metrics)}. "
                f"Choose from: {', '.join(sorted(SMOKE_METRICS))}."
            )
        selected_metrics = [SMOKE_METRICS[name] for name in metric_names]
        evaluator = RAGASEvaluator()
        evaluator.rag_metrics = selected_metrics
        result = evaluator.evaluate(
            rag_responses=[rag],
            ground_truths=["Smoke test ground truth placeholder."],
            experiment_name="live_smoke",
            sample_ids=["live-smoke-001"],
            sample_metadata=[{
                "task_type": "smoke_test",
                "difficulty": "simple",
                "ground_truth_points": [],
            }],
            run_metadata={
                "snapshot_date": snapshot_date,
                "active_setup": config["data"].get("active_setup", "default"),
                "smoke_test": True,
            },
        )
        print(json.dumps({
            "evaluation_output_path": result["output_path"],
            "metrics_used": metric_names,
            "metrics": result["metrics"],
        }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
