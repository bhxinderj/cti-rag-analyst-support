"""
RAGAS Evaluation Module.

Evaluates the RAG pipeline with RAGAS and stores trace-rich run artifacts.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from ..rag.chain import RAGResponse
from ..utils.config import get_project_root, load_config

logger = logging.getLogger(__name__)

# Compatibility shims for tests and earlier artifact-building code paths.
SingleTurnSample = object
EvaluationDataset = object

_CITATION_BLOCK_RE = re.compile(r"\[(?:Source|Sources)\s*:\s*([^\]]+)\]")
_NON_CLAIM_LINES = {
    "Summary:",
    "Why it matters:",
    "Recommended actions / Mitigations:",
    "Evidence:",
    "Unknowns / Gaps:",
    "Missing:",
    "Source-grounded summary could not be preserved after citation checks.",
    "No clearly source-grounded impact statement could be preserved from the generated answer.",
    "No reliable mitigation guidance is present in the retrieved context.",
    "No clearly source-grounded evidence statements could be preserved from the generated answer.",
}


def _results_records(results) -> list[dict]:
    """Convert RAGAS results to plain per-sample records."""
    if not hasattr(results, "to_pandas"):
        return []
    frame = results.to_pandas()
    if hasattr(frame, "to_dict"):
        return frame.to_dict(orient="records")
    return []


def _aggregate_metric_means(records: list[dict], metrics: list) -> dict[str, float]:
    """Aggregate mean metric values from per-sample result records."""
    aggregated: dict[str, float] = {}
    for metric in metrics:
        values = [
            float(record[metric.name])
            for record in records
            if isinstance(record.get(metric.name), (int, float))
        ]
        if values:
            aggregated[metric.name] = sum(values) / len(values)
    return aggregated


def _build_grounding_stats(response: RAGResponse) -> dict:
    """Extract lightweight grounding transparency metrics from a response."""
    cited_lines = 0
    uncited_lines = 0

    for line in response.answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped in _NON_CLAIM_LINES:
            continue
        if stripped.startswith("- Grounding note:"):
            continue

        if _CITATION_BLOCK_RE.search(line):
            cited_lines += 1
        else:
            uncited_lines += 1

    total_claim_lines = cited_lines + uncited_lines
    cited_doc_ids = [
        doc.get("doc_id", "")
        for doc in response.source_documents
        if doc.get("cited_in_answer") and doc.get("doc_id")
    ]
    retrieved_doc_ids = [
        doc.get("doc_id", "")
        for doc in response.source_documents
        if doc.get("doc_id")
    ]

    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "cited_doc_ids": cited_doc_ids,
        "used_chunks": [
            doc for doc in response.source_documents
            if doc.get("cited_in_answer")
        ],
        "num_cited_lines": cited_lines,
        "num_uncited_lines": uncited_lines,
        "citation_coverage_ratio": (
            cited_lines / total_claim_lines if total_claim_lines else 0.0
        ),
        "grounding_warnings": list(response.grounding_warnings),
        "abstention_reason": response.abstention_reason,
    }


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.
    """

    def __init__(self):
        config = load_config()
        eval_config = config["evaluation"]["ragas"]
        llm_config = config["llm"]
        emb_config = config["embedding"]

        eval_model = eval_config.get("eval_llm", llm_config["model_name"])
        self.eval_llm = LangchainLLMWrapper(
            ChatOllama(
                model=eval_model,
                base_url=llm_config["base_url"],
                temperature=0.0,
            )
        )

        self.eval_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=emb_config["model_name"],
                model_kwargs={"device": emb_config["device"]},
            )
        )

        self.rag_metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
        ]
        self.baseline_metrics = [
            answer_relevancy,
            answer_correctness,
        ]

        logger.info("RAGASEvaluator initialized (eval_llm=%s)", eval_model)

        self.results_dir = get_project_root() / config["evaluation"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_run_artifacts(
        self,
        responses: list[RAGResponse],
        ground_truths: list[str],
        experiment_name: str,
        sample_ids: list[str] | None = None,
        sample_metadata: list[dict] | None = None,
        run_metadata: dict | None = None,
        result_prefix: str = "ragas",
        metrics_used: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        ragas_records: list[dict] | None = None,
        is_baseline: bool = False,
    ) -> dict:
        """Persist a trace-rich evaluation artifact JSON."""
        per_sample: list[dict] = []
        sample_metadata = sample_metadata or []
        ragas_records = ragas_records or []

        for index, response in enumerate(responses):
            grounding_stats = _build_grounding_stats(response)
            sample = {
                "question": response.query,
                "answer": response.answer,
                "ground_truth": ground_truths[index],
                "retrieval_mode": response.retrieval_mode,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
                "num_contexts": len(response.contexts),
                "prompt_context": response.prompt_context,
                "retrieval_trace": response.retrieval_trace,
                **grounding_stats,
            }

            if sample_ids and index < len(sample_ids):
                sample["sample_id"] = sample_ids[index]
            if index < len(sample_metadata):
                sample.update(sample_metadata[index])
            if index < len(ragas_records):
                metric_record = {
                    key: float(value)
                    for key, value in ragas_records[index].items()
                    if isinstance(value, (int, float))
                }
                if metric_record:
                    sample["ragas_metrics"] = metric_record

            per_sample.append(sample)

        timing = {
            "avg_retrieval_time_ms": (
                sum(response.retrieval_time_ms for response in responses) / len(responses)
                if responses else 0.0
            ),
            "avg_generation_time_ms": (
                sum(response.generation_time_ms for response in responses) / len(responses)
                if responses else 0.0
            ),
            "avg_total_time_ms": (
                sum(response.total_time_ms for response in responses) / len(responses)
                if responses else 0.0
            ),
        }

        result_dict = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(responses),
            "retrieval_mode": (
                "baseline" if is_baseline else responses[0].retrieval_mode if responses else "unknown"
            ),
            "is_baseline": is_baseline,
            "metrics_used": metrics_used or [],
            "metrics": metrics or {},
            "run_metadata": run_metadata or {},
            "timing": timing,
            "per_sample": per_sample,
        }

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"{result_prefix}_{experiment_name}_{timestamp_str}.json"
        result_dict["output_path"] = str(output_path)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(result_dict, handle, indent=2, default=str)

        logger.info("Evaluation artifacts saved: %s", output_path)
        return result_dict

    def evaluate(
        self,
        rag_responses: list[RAGResponse],
        ground_truths: list[str],
        experiment_name: str = "default",
        query_metadata: list[dict] | None = None,
        is_baseline: bool = False,
        sample_ids: list[str] | None = None,
        sample_metadata: list[dict] | None = None,
        run_metadata: dict | None = None,
    ) -> dict:
        """Run RAGAS evaluation and persist enriched artifacts."""
        if len(rag_responses) != len(ground_truths):
            raise ValueError(
                f"Mismatch: {len(rag_responses)} responses vs {len(ground_truths)} ground truths"
            )

        baseline_metrics = getattr(self, "baseline_metrics", [answer_relevancy, answer_correctness])
        rag_metrics = getattr(
            self,
            "rag_metrics",
            [faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
        )
        metrics = baseline_metrics if is_baseline else rag_metrics
        mode_label = (
            "baseline"
            if is_baseline
            else rag_responses[0].retrieval_mode if rag_responses else "unknown"
        )
        sample_metadata = sample_metadata or query_metadata or []

        eval_data = {
            "question": [response.query for response in rag_responses],
            "answer": [response.answer for response in rag_responses],
            "contexts": [response.contexts if response.contexts else ["N/A"] for response in rag_responses],
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(eval_data)

        logger.info(
            "Running RAGAS evaluation on %d samples (mode=%s, metrics=%s)",
            len(dataset),
            mode_label,
            [metric.name for metric in metrics],
        )

        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
        )

        ragas_records = _results_records(results)
        aggregated_metrics = _aggregate_metric_means(ragas_records, metrics)

        return self.save_run_artifacts(
            responses=rag_responses,
            ground_truths=ground_truths,
            experiment_name=experiment_name,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            run_metadata=run_metadata,
            result_prefix="ragas",
            metrics_used=[metric.name for metric in metrics],
            metrics=aggregated_metrics,
            ragas_records=ragas_records,
            is_baseline=is_baseline,
        )
