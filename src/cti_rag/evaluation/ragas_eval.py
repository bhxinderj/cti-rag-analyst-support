"""
RAGAS Evaluation Module.

Evaluates the RAG pipeline using the RAGAS framework with four core metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved chunks relevant and well-ranked?
- Context Recall: Does the retrieved context cover the ground truth?

Baseline mode:
  When is_baseline=True, only answer_relevancy is computed (the other three
  metrics require non-empty retrieval contexts). This is the methodologically
  correct approach: retrieval-dependent metrics characterize the RAG pipeline
  alone, while answer_relevancy enables direct RAG-vs-baseline comparison.

The evaluator uses the same local Ollama LLM as the generator.
This is a known limitation documented in thesis section 4.3.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..rag.chain import RAGResponse
from ..utils.config import load_config, get_project_root

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS metrics.

    Usage:
        evaluator = RAGASEvaluator()
        # RAG evaluation (all 4 metrics)
        results = evaluator.evaluate(rag_responses, ground_truths, query_metadata=metadata)
        # Baseline evaluation (answer_relevancy only)
        results = evaluator.evaluate(baseline_responses, ground_truths, is_baseline=True)
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

        self.rag_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Baseline: only answer_relevancy (no contexts available)
        self.baseline_metrics = [
            answer_relevancy,
        ]

        self.results_dir = get_project_root() / config["evaluation"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        rag_responses: list[RAGResponse],
        ground_truths: list[str],
        experiment_name: str = "default",
        query_metadata: list[dict] | None = None,
        is_baseline: bool = False,
    ) -> dict:
        """
        Run RAGAS evaluation on a set of RAG responses.

        Args:
            rag_responses: List of RAGResponse objects from the pipeline
            ground_truths: List of ground truth answers (one per query)
            experiment_name: Name for this evaluation run
            query_metadata: Optional list of dicts with {id, task_type, difficulty}
            is_baseline: If True, only compute answer_relevancy (no context-dependent metrics)

        Returns:
            Dictionary with metric scores and per-sample results
        """
        if len(rag_responses) != len(ground_truths):
            raise ValueError(
                f"Mismatch: {len(rag_responses)} responses vs {len(ground_truths)} ground truths"
            )

        metrics = self.baseline_metrics if is_baseline else self.rag_metrics
        mode_label = "baseline" if is_baseline else rag_responses[0].retrieval_mode if rag_responses else "unknown"

        # Build RAGAS dataset
        # For baseline: contexts are empty lists, but RAGAS Dataset requires the field.
        # answer_relevancy does not use contexts, so empty lists are safe.
        eval_data = {
            "question": [r.query for r in rag_responses],
            "answer": [r.answer for r in rag_responses],
            "contexts": [r.contexts if r.contexts else ["N/A"] for r in rag_responses],
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(eval_data)

        logger.info(
            f"Running RAGAS evaluation on {len(dataset)} samples "
            f"(mode={mode_label}, metrics={[m.name for m in metrics]})"
        )

        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
        )

        # Build per-sample results with metadata
        per_sample = []
        results_df = results.to_pandas() if hasattr(results, "to_pandas") else None

        for i, resp in enumerate(rag_responses):
            sample = {
                "question": resp.query,
                "answer": resp.answer,
                "ground_truth": ground_truths[i],
                "retrieval_mode": resp.retrieval_mode,
                "retrieval_time_ms": resp.retrieval_time_ms,
                "generation_time_ms": resp.generation_time_ms,
                "num_contexts": len(resp.contexts),
            }

            # Add query metadata if provided
            if query_metadata and i < len(query_metadata):
                sample["query_id"] = query_metadata[i].get("id", f"q_{i}")
                sample["task_type"] = query_metadata[i].get("task_type", "unknown")
                sample["difficulty"] = query_metadata[i].get("difficulty", "unknown")

            # Add per-sample metric scores from RAGAS
            if results_df is not None and i < len(results_df):
                for m in metrics:
                    col = m.name
                    if col in results_df.columns:
                        sample[col] = float(results_df.iloc[i][col])

            # Add source document IDs for traceability
            if resp.source_documents:
                sample["source_doc_ids"] = [d.get("doc_id", "") for d in resp.source_documents]

            per_sample.append(sample)

        # Build result dict
        result_dict = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(rag_responses),
            "retrieval_mode": mode_label,
            "is_baseline": is_baseline,
            "metrics_used": [m.name for m in metrics],
            "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
            "per_sample": per_sample,
        }

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"ragas_{experiment_name}_{timestamp_str}.json"
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"RAGAS results saved: {output_path}")
        logger.info(f"Metrics: {result_dict['metrics']}")

        return result_dict
