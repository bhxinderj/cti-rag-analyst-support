import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.cti_rag.evaluation.ragas_eval import RAGASEvaluator, _ragas_answer_text
from src.cti_rag.rag.chain import RAGResponse


def test_ragas_answer_text_prefers_l2_output_for_templated_responses():
    templated = RAGResponse(
        query="q",
        answer="## L1 Card\n\nCVSS: 9.8\n\nNarrative text [Source: nvd_x].",
        contexts=["c"],
        source_documents=[],
        l2_output="Narrative text [Source: nvd_x].",
    )
    legacy = RAGResponse(
        query="q",
        answer="Summary: plain answer [Source: nvd_x].",
        contexts=["c"],
        source_documents=[],
    )
    assert _ragas_answer_text(templated) == "Narrative text [Source: nvd_x]."
    assert _ragas_answer_text(legacy) == "Summary: plain answer [Source: nvd_x]."


class _FakeSeries:
    def __init__(self, values):
        self._values = [value for value in values if value is not None]

    def dropna(self):
        return self

    def __len__(self):
        return len(self._values)

    def mean(self):
        return sum(self._values) / len(self._values)


class _FakeDataFrame:
    def __init__(self, records):
        self._records = records
        self.columns = list(records[0].keys()) if records else []

    def __getitem__(self, key):
        return _FakeSeries([record.get(key) for record in self._records])

    def to_dict(self, orient="records"):
        assert orient == "records"
        return list(self._records)


class _FakeResults:
    scores = True

    def __init__(self, records):
        self._records = records

    def to_pandas(self):
        return _FakeDataFrame(self._records)


def test_evaluator_writes_trace_rich_result_json():
    response = RAGResponse(
        query="What is CVE-2024-3094?",
        answer="Summary: XZ Utils backdoor. [Source: XZ Utils vulnerability | nvd_CVE-2024-3094]",
        contexts=["CVE-2024-3094 is the XZ Utils backdoor."],
        source_documents=[
            {
                "doc_id": "nvd_CVE-2024-3094",
                "title": "XZ Utils vulnerability",
                "score": 0.91,
                "rank": 1,
                "citation_label": "XZ Utils vulnerability | nvd_CVE-2024-3094",
                "cited_in_answer": True,
            },
            {
                "doc_id": "cisa_kev_CVE-2021-44228",
                "title": "Log4Shell",
                "score": 0.12,
                "rank": 2,
                "citation_label": "Log4Shell | cisa_kev_CVE-2021-44228",
                "cited_in_answer": False,
            },
        ],
        retrieval_time_ms=10.0,
        generation_time_ms=20.0,
        total_time_ms=30.0,
        retrieval_mode="hybrid",
        prompt_context="[1] Source ID: nvd_CVE-2024-3094",
        retrieval_trace={
            "query": "What is CVE-2024-3094?",
            "mode": "hybrid",
            "stages": {
                "bm25": [{"doc_id": "nvd_CVE-2024-3094", "rank": 1, "score": 12.0}],
                "final": [{"doc_id": "nvd_CVE-2024-3094", "rank": 1, "score": 0.91}],
            },
        },
    )

    fake_results = _FakeResults([
        {
            "user_input": response.query,
            "response": response.answer,
            "retrieved_contexts": response.contexts,
            "reference": "CVE-2024-3094 is the XZ Utils backdoor.",
            "faithfulness": 1.0,
            "answer_relevancy": 0.5,
        }
    ])

    evaluator = object.__new__(RAGASEvaluator)
    evaluator.metrics = []
    evaluator.eval_llm = None
    evaluator.eval_embeddings = None

    with TemporaryDirectory() as tmp_dir:
        evaluator.results_dir = Path(tmp_dir)

        with patch("src.cti_rag.evaluation.ragas_eval.evaluate", return_value=fake_results), patch(
            "src.cti_rag.evaluation.ragas_eval.SingleTurnSample",
            side_effect=lambda **kwargs: kwargs,
        ), patch(
            "src.cti_rag.evaluation.ragas_eval.EvaluationDataset",
            side_effect=lambda samples: samples,
        ):
            result = evaluator.evaluate(
                rag_responses=[response],
                ground_truths=["CVE-2024-3094 is the XZ Utils backdoor."],
                experiment_name="smoke",
                sample_ids=["eval-cve-3094"],
                sample_metadata=[{
                    "id": "eval-cve-3094",
                    "task_type": "vulnerability_analysis",
                    "difficulty": "simple",
                    "ground_truth_points": [
                        "CVE-2024-3094 is the XZ Utils backdoor.",
                        "It affected SSH authentication.",
                    ],
                }],
                run_metadata={"snapshot_date": "2026-03-28"},
            )

        saved_files = list(Path(tmp_dir).glob("ragas_smoke_*.json"))
        assert len(saved_files) == 1

        saved = json.loads(saved_files[0].read_text())
        assert result["metrics"]["faithfulness"] == 1.0
        assert result["run_metadata"]["snapshot_date"] == "2026-03-28"
        assert saved["per_sample"][0]["sample_id"] == "eval-cve-3094"
        assert saved["per_sample"][0]["task_type"] == "vulnerability_analysis"
        assert saved["per_sample"][0]["difficulty"] == "simple"
        assert saved["per_sample"][0]["ground_truth_points"] == [
            "CVE-2024-3094 is the XZ Utils backdoor.",
            "It affected SSH authentication.",
        ]
        assert saved["per_sample"][0]["retrieved_doc_ids"] == [
            "nvd_CVE-2024-3094",
            "cisa_kev_CVE-2021-44228",
        ]
        assert saved["per_sample"][0]["cited_doc_ids"] == ["nvd_CVE-2024-3094"]
        assert saved["per_sample"][0]["used_chunks"][0]["rank"] == 1
        assert saved["per_sample"][0]["retrieval_trace"]["stages"]["bm25"][0]["doc_id"] == "nvd_CVE-2024-3094"
        assert "avg_generation_time_ms" in saved["timing"]
        assert saved["output_path"].endswith(".json")


def test_evaluator_can_save_baseline_artifacts_without_ragas_metrics():
    response = RAGResponse(
        query="What is CVE-2024-3094?",
        answer="Summary: CVE-2024-3094 is the XZ Utils backdoor.",
        generation_time_ms=15.0,
        total_time_ms=15.0,
        retrieval_mode="baseline_no_retrieval",
    )

    evaluator = object.__new__(RAGASEvaluator)
    with TemporaryDirectory() as tmp_dir:
        evaluator.results_dir = Path(tmp_dir)
        result = evaluator.save_run_artifacts(
            responses=[response],
            ground_truths=["CVE-2024-3094 is the XZ Utils backdoor."],
            experiment_name="baseline_smoke",
            sample_ids=["eval-cve-3094"],
            sample_metadata=[{
                "task_type": "vulnerability_analysis",
                "difficulty": "simple",
                "ground_truth_points": ["CVE-2024-3094 is the XZ Utils backdoor."],
            }],
            run_metadata={"snapshot_date": "2026-03-28"},
            result_prefix="baseline",
        )

        saved_files = list(Path(tmp_dir).glob("baseline_baseline_smoke_*.json"))
        assert len(saved_files) == 1
        assert result["metrics"] == {}
        assert result["per_sample"][0]["retrieval_mode"] == "baseline_no_retrieval"
        assert "ragas_metrics" not in result["per_sample"][0]
