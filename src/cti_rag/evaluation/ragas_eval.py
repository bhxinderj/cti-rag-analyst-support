"""
RAGAS Evaluation Module.

Evaluates the RAG pipeline using the RAGAS framework with four core metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved chunks relevant and well-ranked?
- Context Recall: Does the retrieved context cover the ground truth?

The evaluator uses the same local Ollama LLM as the generator.
This is a known limitation documented in thesis section 4.3.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ..rag.chain import RAGResponse
from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.

    Usage:
        evaluator = RAGASEvaluator()
        results = evaluator.evaluate(rag_responses, ground_truths)
    """

    def __init__(self):
        config = load_config()
        llm_config = config["llm"]
        emb_config = config["embedding"]

        # Use same Ollama LLM as evaluator
        self.eval_llm = LangchainLLMWrapper(
            ChatOllama(
                model=llm_config["model_name"],
                base_url=llm_config["base_url"],
                temperature=0.0,  # Deterministic for evaluation
            )
        )

        # Use same embedding model
        self.eval_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=emb_config["model_name"],
                model_kwargs={"device": emb_config["device"]},
            )
        )

        # RAGAS 0.2.x: instantiate metric classes
        self.metrics = [
            Faithfulness(llm=self.eval_llm),
            AnswerRelevancy(llm=self.eval_llm, embeddings=self.eval_embeddings),
            LLMContextPrecisionWithReference(llm=self.eval_llm),
            LLMContextRecall(llm=self.eval_llm),
        ]

        self.results_dir = get_project_root() / config["evaluation"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_response_trace(
        resp: RAGResponse,
        ground_truth: str,
        sample_id: str | None = None,
    ) -> dict:
        """Serialize one pipeline run for later inspection and analysis."""
        used_chunks = list(resp.source_documents)
        return {
            "sample_id": sample_id,
            "query": resp.query,
            "ground_truth": ground_truth,
            "answer": resp.answer,
            "retrieval_mode": resp.retrieval_mode,
            "retrieval_time_ms": resp.retrieval_time_ms,
            "generation_time_ms": resp.generation_time_ms,
            "total_time_ms": resp.total_time_ms,
            "abstention_reason": resp.abstention_reason,
            "prompt_context": resp.prompt_context,
            "retrieved_contexts": resp.contexts if resp.contexts else ["No context retrieved."],
            "retrieved_doc_ids": [chunk.get("doc_id") for chunk in used_chunks],
            "used_chunks": used_chunks,
            "cited_doc_ids": [
                chunk.get("doc_id")
                for chunk in used_chunks
                if chunk.get("cited_in_answer")
            ],
            "retrieval_trace": dict(resp.retrieval_trace),
        }

    def evaluate(
        self,
        rag_responses: list[RAGResponse],
        ground_truths: list[str],
        experiment_name: str = "default",
        sample_ids: list[str] | None = None,
    ) -> dict:
        """
        Run RAGAS evaluation on a set of RAG responses.

        Args:
            rag_responses: List of RAGResponse objects from the pipeline
            ground_truths: List of ground truth answers (one per query)
            experiment_name: Name for this evaluation run

        Returns:
            Dictionary with metric scores and per-sample results
        """
        if len(rag_responses) != len(ground_truths):
            raise ValueError(
                f"Mismatch: {len(rag_responses)} responses vs {len(ground_truths)} ground truths"
            )
        if sample_ids is not None and len(sample_ids) != len(rag_responses):
            raise ValueError(
                f"Mismatch: {len(sample_ids)} sample IDs vs {len(rag_responses)} responses"
            )

        # Build RAGAS 0.2.x EvaluationDataset using SingleTurnSample
        samples = []
        for resp, gt in zip(rag_responses, ground_truths):
            sample = SingleTurnSample(
                user_input=resp.query,
                response=resp.answer,
                retrieved_contexts=resp.contexts if resp.contexts else ["No context retrieved."],
                reference=gt,
            )
            samples.append(sample)

        dataset = EvaluationDataset(samples=samples)

        logger.info(f"Running RAGAS evaluation on {len(dataset)} samples...")

        # Run evaluation with extended timeout for local Ollama model.
        # max_workers=1 because Ollama processes requests sequentially.
        run_config = RunConfig(
            timeout=600,        # 10 min per LLM call (8B model is slow on complex prompts)
            max_retries=3,
            max_workers=1,      # Sequential: local Ollama cannot parallelize
            seed=42,            # Reproducibility
        )

        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True,
        )

        # Extract aggregate metrics
        metrics_dict = {}
        if hasattr(results, "scores"):
            # RAGAS 0.2.x returns scores as list of dicts
            scores_df = results.to_pandas()
            for col in scores_df.columns:
                if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                    values = scores_df[col].dropna()
                    if len(values) > 0:
                        metrics_dict[col] = float(values.mean())
        else:
            # Fallback: iterate result dict
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    metrics_dict[k] = float(v)

        # Build per-sample results
        per_sample_metric_records = []
        if hasattr(results, "to_pandas"):
            per_sample_metric_records = results.to_pandas().to_dict(orient="records")

        per_sample = []
        for idx, (resp, gt) in enumerate(zip(rag_responses, ground_truths)):
            trace = self.build_response_trace(
                resp=resp,
                ground_truth=gt,
                sample_id=sample_ids[idx] if sample_ids is not None else None,
            )
            raw_metrics = per_sample_metric_records[idx] if idx < len(per_sample_metric_records) else {}
            metric_values = {
                key: value
                for key, value in raw_metrics.items()
                if key not in ("user_input", "response", "retrieved_contexts", "reference")
            }
            per_sample.append({
                **trace,
                "ragas_metrics": metric_values,
            })

        # Save results
        result_dict = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(rag_responses),
            "retrieval_mode": rag_responses[0].retrieval_mode if rag_responses else "unknown",
            "metrics": metrics_dict,
            "per_sample": per_sample,
        }

        output_path = self.results_dir / f"ragas_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"RAGAS results saved: {output_path}")
        logger.info(f"Metrics: {metrics_dict}")

        return result_dict
