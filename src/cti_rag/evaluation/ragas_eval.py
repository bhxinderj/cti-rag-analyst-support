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

        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        self.results_dir = get_project_root() / config["evaluation"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        rag_responses: list[RAGResponse],
        ground_truths: list[str],
        experiment_name: str = "default",
    ) -> dict:
        """
        Run RAGAS evaluation on a set of RAG responses.

        Args:
            rag_responses: List of RAGResponse objects from the pipeline
            ground_truths: List of ground truth answers (one per query)
            experiment_name: Name for this evaluation run

        Returns:
            Dictionary with metric scores
        """
        if len(rag_responses) != len(ground_truths):
            raise ValueError(
                f"Mismatch: {len(rag_responses)} responses vs {len(ground_truths)} ground truths"
            )

        # Build RAGAS dataset
        eval_data = {
            "question": [r.query for r in rag_responses],
            "answer": [r.answer for r in rag_responses],
            "contexts": [r.contexts for r in rag_responses],
            "ground_truth": ground_truths,
        }

        dataset = Dataset.from_dict(eval_data)

        logger.info(f"Running RAGAS evaluation on {len(dataset)} samples...")

        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
        )

        # Save results
        result_dict = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(rag_responses),
            "retrieval_mode": rag_responses[0].retrieval_mode if rag_responses else "unknown",
            "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
            "per_sample": results.to_pandas().to_dict(orient="records") if hasattr(results, "to_pandas") else [],
        }

        output_path = self.results_dir / f"ragas_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"RAGAS results saved: {output_path}")
        logger.info(f"Metrics: {result_dict['metrics']}")

        return result_dict
