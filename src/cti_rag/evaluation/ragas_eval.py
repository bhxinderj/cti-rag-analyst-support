"""
RAGAS Evaluation Module.

Evaluates the RAG pipeline with RAGAS and stores trace-rich run artifacts.
"""

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_huggingface import HuggingFaceEmbeddings
import torch

from ..rag.chain import RAGResponse
from ..utils.config import (
    get_project_root,
    load_config,
    require_local_hf_snapshot,
    suppress_noisy_third_party_logs,
)
from ._judge_llm import build_judge_llm

logger = logging.getLogger(__name__)
suppress_noisy_third_party_logs()

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


def _resolve_embedding_device(requested_device: str) -> str:
    """Fall back to CPU when the configured accelerator is unavailable."""
    if requested_device != "mps":
        return requested_device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return requested_device

    logger.warning("Configured embedding device 'mps' is unavailable on this host. Falling back to 'cpu'.")
    return "cpu"


def _results_records(results) -> list[dict]:
    """Convert RAGAS results to plain per-sample records."""
    if not hasattr(results, "to_pandas"):
        return []
    frame = results.to_pandas()
    if hasattr(frame, "to_dict"):
        return frame.to_dict(orient="records")
    return []


def _aggregate_metric_means(records: list[dict], metrics: list) -> dict[str, float]:
    """Aggregate mean metric values from per-sample result records.

    NaN values (produced when the eval LLM fails to parse a single sample)
    are excluded so they do not propagate into the aggregate mean.
    """
    aggregated: dict[str, float] = {}
    for metric in metrics:
        values: list[float] = []
        nan_count = 0
        for record in records:
            raw = record.get(metric.name)
            if not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if math.isnan(value):
                nan_count += 1
                continue
            values.append(value)
        if nan_count:
            logger.warning(
                "Metric %s: excluded %d NaN sample(s) from aggregate (%d valid)",
                metric.name,
                nan_count,
                len(values),
            )
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

    @staticmethod
    def _build_eval_llm(provider: str, eval_config: dict, ollama_base_url: str):
        """Instantiate the configured RAGAS evaluation LLM.

        Delegates to :func:`build_judge_llm` and wraps the result in
        ``LangchainLLMWrapper`` so RAGAS can call it. Ollama callers need
        the longer 600 s timeout because RAGAS issues many judgments
        concurrently — that is hard-coded here and should not be tuned
        down without also reducing ``RunConfig.max_workers``.

        Kept as a static method for backwards compatibility with callers
        and tests that patch ``RAGASEvaluator._build_eval_llm``.
        """
        model_label, raw_llm = build_judge_llm(
            provider=provider,
            eval_config=eval_config,
            ollama_base_url=ollama_base_url,
            timeout=600,
        )
        return model_label, LangchainLLMWrapper(raw_llm)

    def __init__(self):
        config = load_config()
        eval_config = config["evaluation"]["ragas"]
        llm_config = config["llm"]
        emb_config = config["embedding"]

        provider = eval_config.get("eval_llm_provider", "ollama").lower()
        self.eval_model_label, self.eval_llm = self._build_eval_llm(
            provider=provider,
            eval_config=eval_config,
            ollama_base_url=llm_config["base_url"],
        )
        logger.info("RAGAS eval LLM: %s", self.eval_model_label)

        self.eval_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=str(
                    require_local_hf_snapshot(
                        emb_config["model_name"],
                        artifact_label="Embedding model",
                    )
                ),
                model_kwargs={"device": _resolve_embedding_device(emb_config["device"])},
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

        # Local Ollama needs conservative concurrency; hosted APIs can
        # handle the RAGAS defaults (much more parallelism).
        if provider == "ollama":
            self.run_config = RunConfig(
                timeout=600,
                max_workers=2,
                max_retries=10,
                max_wait=180,
            )
        else:
            self.run_config = RunConfig(
                timeout=180,
                max_workers=8,
                max_retries=6,
                max_wait=60,
            )

        logger.info("RAGASEvaluator initialized (eval_llm=%s)", self.eval_model_label)

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
                "raw_llm_output": getattr(response, "raw_llm_output", ""),
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

        # Stamp the judge model into the artifact so runs stay identifiable.
        run_metadata = dict(run_metadata or {})
        run_metadata.setdefault("eval_llm", getattr(self, "eval_model_label", "unknown"))

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

        ragas_records: list[dict] = []
        aggregated_metrics: dict[str, float] = {}
        metrics_used = [metric.name for metric in metrics]
        result_prefix = "ragas"

        # ``run_config`` is set by ``__init__`` but some tests instantiate
        # the evaluator via ``object.__new__`` to skip heavy setup — fall
        # back to RAGAS defaults in that case. A crash inside RAGAS must
        # not lose the retrieval-trace artifacts, hence the partial save.
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.eval_llm,
                embeddings=self.eval_embeddings,
                run_config=getattr(self, "run_config", None),
                raise_exceptions=False,
            )

            ragas_records = _results_records(results)
            aggregated_metrics = _aggregate_metric_means(ragas_records, metrics)
        except Exception:
            logger.exception(
                "RAGAS evaluate() crashed — saving retrieval artifacts without metric scores."
            )
            result_prefix = "ragas_partial"

        return self.save_run_artifacts(
            responses=rag_responses,
            ground_truths=ground_truths,
            experiment_name=experiment_name,
            sample_ids=sample_ids,
            sample_metadata=sample_metadata,
            run_metadata=run_metadata,
            result_prefix=result_prefix,
            metrics_used=metrics_used,
            metrics=aggregated_metrics,
            ragas_records=ragas_records,
            is_baseline=is_baseline,
        )
