import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from main import _load_eval_queries


class TestEvalQueryLoading(unittest.TestCase):
    def test_load_eval_queries_builds_reference_from_ground_truth_points(self):
        with TemporaryDirectory() as tmp_dir:
            query_file = Path(tmp_dir) / "eval_queries.yaml"
            query_file.write_text(
                """
queries:
  - id: sample_001
    question: "What is CVE-2024-3094?"
    task_type: vulnerability_analysis
    difficulty: simple
    ground_truth_points:
      - "CVE-2024-3094 is an XZ Utils backdoor."
      - "It affects SSH authentication."
""",
                encoding="utf-8",
            )

            queries = _load_eval_queries(query_file)

        self.assertEqual(len(queries), 1)
        self.assertEqual(
            queries[0]["ground_truth"],
            "CVE-2024-3094 is an XZ Utils backdoor. It affects SSH authentication.",
        )
        self.assertEqual(
            queries[0]["ground_truth_points"],
            [
                "CVE-2024-3094 is an XZ Utils backdoor.",
                "It affects SSH authentication.",
            ],
        )

    def test_load_eval_queries_rejects_duplicate_ids(self):
        with TemporaryDirectory() as tmp_dir:
            query_file = Path(tmp_dir) / "eval_queries.yaml"
            query_file.write_text(
                """
queries:
  - id: duplicate_id
    question: "Q1"
    task_type: vulnerability_analysis
    difficulty: simple
    ground_truth: "A1"
  - id: duplicate_id
    question: "Q2"
    task_type: ioc_enrichment
    difficulty: moderate
    ground_truth: "A2"
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate evaluation query id"):
                _load_eval_queries(query_file)


if __name__ == "__main__":
    unittest.main()
