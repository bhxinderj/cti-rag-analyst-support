"""
Unit tests for the Structured Rubric LLM-as-Judge evaluator.

Strategy
--------
The judge LLM is replaced with a hand-written fake that yields deterministic
JSON payloads. No network, no Ollama, no OpenRouter. The tests exercise:

- JSON parsing (clean, nested-dict, flat-int, out-of-range, malformed,
  wrapped-in-prose, markdown-fenced)
- Per-dimension score clamping
- ``RubricScore.total`` and missing-dimension propagation
- Aggregation: overall means, per-template, per-task_type, per-difficulty
- Offline loader: ``load_samples_from_ragas_artifact`` round-trips the
  schema produced by ``RAGASEvaluator.save_run_artifacts``
- ``RubricEvaluator.score_sample`` end-to-end with a fake judge

Follows the repo convention: plain ``def test_*`` functions with bare
``assert`` statements, discovered by ``tests/run_smoke_checks.py`` or by
importlib reflection.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.cti_rag.evaluation import rubric_eval
from src.cti_rag.evaluation.rubric_eval import (
    DIMENSIONS,
    RubricEvaluator,
    RubricSample,
    RubricSampleResult,
    RubricScore,
    _coerce_score,
    _extract_json_blob,
    _parse_judge_payload,
    aggregate_rubric_scores,
    load_samples_from_ragas_artifact,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeJudge:
    """Returns a canned response per ``invoke`` call, cycling through a list."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._responses:
            return _FakeMessage("")
        payload = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        return _FakeMessage(payload)


def _build_evaluator_with_fake(judge: _FakeJudge) -> RubricEvaluator:
    """Return a RubricEvaluator whose judge is replaced by the fake.

    We bypass ``__init__`` so the test never touches config / network.
    """
    evaluator = RubricEvaluator.__new__(RubricEvaluator)
    evaluator.judge_llm = judge
    evaluator.judge_model_label = "fake:unit-test"
    evaluator.results_dir = Path("/tmp")
    evaluator._provider = "fake"
    return evaluator


def _make_payload(p: int = 2, a: int = 1, c: int = 3, t: int = 2) -> str:
    return json.dumps(
        {
            "prioritization": {"score": p, "justification": "p-just"},
            "actionability": {"score": a, "justification": "a-just"},
            "completeness": {"score": c, "justification": "c-just"},
            "traceability": {"score": t, "justification": "t-just"},
        }
    )


# ---------------------------------------------------------------------------
# _extract_json_blob / _coerce_score
# ---------------------------------------------------------------------------


def test_extract_json_blob_strict_parses_pure_json():
    payload = '{"a": 1}'
    assert _extract_json_blob(payload) == payload


def test_extract_json_blob_strips_surrounding_prose():
    payload = 'Sure, here is the score:\n```json\n{"a":1,"b":2}\n```\nThanks!'
    extracted = _extract_json_blob(payload)
    assert extracted is not None
    assert json.loads(extracted) == {"a": 1, "b": 2}


def test_extract_json_blob_returns_none_on_garbage():
    assert _extract_json_blob("no braces here") is None
    assert _extract_json_blob("") is None


def test_coerce_score_happy_path():
    assert _coerce_score(2) == 2
    assert _coerce_score("3") == 3
    assert _coerce_score(1.7) == 2  # rounds


def test_coerce_score_clamps_out_of_range():
    assert _coerce_score(-2) == 0
    assert _coerce_score(99) == 3


def test_coerce_score_returns_none_on_garbage():
    assert _coerce_score("foo") is None
    assert _coerce_score(None) is None


# ---------------------------------------------------------------------------
# _parse_judge_payload
# ---------------------------------------------------------------------------


def test_parse_judge_payload_happy_path():
    scores, justifications, err = _parse_judge_payload(_make_payload(3, 2, 3, 2))
    assert err is None
    assert scores.prioritization == 3
    assert scores.actionability == 2
    assert scores.completeness == 3
    assert scores.traceability == 2
    assert scores.total() == 2.5
    assert justifications == {
        "prioritization": "p-just",
        "actionability": "a-just",
        "completeness": "c-just",
        "traceability": "t-just",
    }


def test_parse_judge_payload_accepts_flat_int_shape():
    """Some judges return {dim: 2} instead of {dim: {"score": 2}}. Tolerate it."""
    payload = json.dumps(
        {"prioritization": 2, "actionability": 1, "completeness": 3, "traceability": 0}
    )
    scores, justifications, err = _parse_judge_payload(payload)
    assert err is None
    assert scores.prioritization == 2
    assert scores.traceability == 0
    assert justifications == {}


def test_parse_judge_payload_flags_missing_dimension():
    payload = json.dumps({"prioritization": {"score": 2}, "actionability": {"score": 1}})
    scores, _, err = _parse_judge_payload(payload)
    assert err is not None
    assert "completeness" in err and "traceability" in err
    assert scores.completeness is None
    assert scores.traceability is None
    # total() returns None when any dimension is missing
    assert scores.total() is None


def test_parse_judge_payload_clamps_out_of_range_scores():
    payload = json.dumps(
        {
            "prioritization": {"score": 7},
            "actionability": {"score": -3},
            "completeness": {"score": 2},
            "traceability": {"score": 2},
        }
    )
    scores, _, err = _parse_judge_payload(payload)
    assert err is None
    assert scores.prioritization == 3
    assert scores.actionability == 0


def test_parse_judge_payload_rejects_non_json():
    scores, _, err = _parse_judge_payload("nope, just words.")
    assert err == "judge response was not valid JSON"
    assert scores.prioritization is None


def test_parse_judge_payload_rejects_non_object_json():
    scores, _, err = _parse_judge_payload("[1, 2, 3]")
    assert err is not None
    assert scores.prioritization is None


# ---------------------------------------------------------------------------
# RubricScore / RubricSampleResult
# ---------------------------------------------------------------------------


def test_rubric_score_total_averages_four_dimensions():
    score = RubricScore(prioritization=3, actionability=3, completeness=3, traceability=3)
    assert score.total() == 3.0
    score = RubricScore(prioritization=0, actionability=2, completeness=2, traceability=0)
    assert score.total() == 1.0


def test_rubric_score_total_is_none_when_any_dimension_missing():
    score = RubricScore(prioritization=3, actionability=None, completeness=3, traceability=3)
    assert score.total() is None


def test_rubric_sample_result_to_dict_round_trips():
    result = RubricSampleResult(
        sample_id="vuln_004",
        template="VulnTriage",
        task_type="vulnerability_analysis",
        difficulty="moderate",
        scores=RubricScore(prioritization=3, actionability=2, completeness=3, traceability=2),
        justifications={"prioritization": "yep"},
        raw_response="...",
        error=None,
    )
    as_dict = result.to_dict()
    assert as_dict["sample_id"] == "vuln_004"
    assert as_dict["scores"]["prioritization"] == 3
    assert as_dict["total"] == 2.5
    assert as_dict["justifications"] == {"prioritization": "yep"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result(
    sample_id: str,
    template: str,
    task_type: str,
    difficulty: str,
    scores: tuple[int, int, int, int],
    error: str | None = None,
) -> RubricSampleResult:
    return RubricSampleResult(
        sample_id=sample_id,
        template=template,
        task_type=task_type,
        difficulty=difficulty,
        scores=RubricScore(*scores),
        error=error,
    )


def test_aggregate_rubric_scores_groups_by_template_and_task_type():
    results = [
        _result("vuln_004", "VulnTriage", "vulnerability_analysis", "moderate", (3, 2, 3, 2)),
        _result("vuln_001", "VulnTriage", "vulnerability_analysis", "simple", (2, 2, 2, 2)),
        _result("ttp_001", "ThreatContext", "ttp_correlation", "moderate", (1, 1, 2, 0)),
        _result("cross_003", "CrossSourceCompare", "cross_source", "complex", (2, 2, 3, 1)),
    ]
    agg = aggregate_rubric_scores(results)
    overall = agg["overall"]
    assert overall["n"] == 4
    assert overall["prioritization_mean"] == 2.0
    assert overall["total_mean"] == round((2.5 + 2.0 + 1.0 + 2.0) / 4, 4)
    assert overall["errors"] == 0

    per_template = agg["per_template"]
    assert set(per_template) == {"VulnTriage", "ThreatContext", "CrossSourceCompare"}
    assert per_template["VulnTriage"]["n"] == 2
    assert per_template["VulnTriage"]["prioritization_mean"] == 2.5

    per_task = agg["per_task_type"]
    assert per_task["vulnerability_analysis"]["n"] == 2
    assert per_task["ttp_correlation"]["traceability_mean"] == 0.0

    per_diff = agg["per_difficulty"]
    assert per_diff["moderate"]["n"] == 2


def test_aggregate_rubric_scores_handles_all_missing_scores():
    results = [
        RubricSampleResult(
            sample_id="x",
            template="VulnTriage",
            task_type="vulnerability_analysis",
            difficulty="simple",
            scores=RubricScore(),
            error="judge failed",
        )
    ]
    agg = aggregate_rubric_scores(results)
    assert agg["overall"]["n"] == 1
    assert agg["overall"]["prioritization_mean"] is None
    assert agg["overall"]["total_mean"] is None
    assert agg["overall"]["errors"] == 1


def test_aggregate_rubric_scores_skips_unknown_groups():
    """Samples without a template still appear in overall but not in per_template."""
    results = [
        _result("vuln_004", "VulnTriage", "vulnerability_analysis", "moderate", (3, 3, 3, 3)),
        RubricSampleResult(
            sample_id="legacy_x",
            template=None,
            task_type=None,
            difficulty=None,
            scores=RubricScore(prioritization=2, actionability=2, completeness=2, traceability=2),
        ),
    ]
    agg = aggregate_rubric_scores(results)
    assert agg["overall"]["n"] == 2
    assert list(agg["per_template"]) == ["VulnTriage"]
    assert agg["per_template"]["VulnTriage"]["n"] == 1


# ---------------------------------------------------------------------------
# Offline loader
# ---------------------------------------------------------------------------


def test_load_samples_from_ragas_artifact_extracts_phase2_records():
    artifact = {
        "experiment_name": "hybrid_templated_phase2_raw_28q",
        "retrieval_mode": "hybrid",
        "num_samples": 2,
        "per_sample": [
            {
                "sample_id": "vuln_004",
                "question": "What is CVE-2023-4966?",
                "answer": "**Triage signal:** CRITICAL ...",
                "ground_truth": "CVE-2023-4966 (Citrix Bleed) ...",
                "template": "VulnTriage",
                "task_type": "vulnerability_analysis",
                "difficulty": "moderate",
            },
            {
                "sample_id": "ttp_001",
                "question": "What initial access techniques ...",
                "answer": "**Threat Actors** ...",
                "ground_truth": "T1190 ...",
                "template": "ThreatContext",
                "task_type": "ttp_correlation",
                "difficulty": "moderate",
            },
        ],
    }
    with TemporaryDirectory() as td:
        path = Path(td) / "ragas_sample.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        samples, meta = load_samples_from_ragas_artifact(path)

    assert len(samples) == 2
    assert samples[0].sample_id == "vuln_004"
    assert samples[0].template == "VulnTriage"
    assert samples[0].task_type == "vulnerability_analysis"
    assert meta["experiment_name"] == "hybrid_templated_phase2_raw_28q"
    assert meta["retrieval_mode"] == "hybrid"


def test_load_samples_from_ragas_artifact_skips_empty_records():
    artifact = {
        "per_sample": [
            {"sample_id": "ok", "question": "Q", "answer": "A"},
            {"sample_id": "empty", "question": "", "answer": "A"},
            {"sample_id": "no_answer", "question": "Q", "answer": ""},
        ]
    }
    with TemporaryDirectory() as td:
        path = Path(td) / "ragas_sample.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        samples, _ = load_samples_from_ragas_artifact(path)

    assert [s.sample_id for s in samples] == ["ok"]


def test_load_samples_from_ragas_artifact_tolerates_legacy_records_without_template():
    """Legacy runs have no ``template`` field; loader should leave it None."""
    artifact = {
        "per_sample": [
            {
                "sample_id": "legacy_x",
                "question": "Q",
                "answer": "A",
                "ground_truth": "GT",
                "task_type": "cross_source",
            }
        ]
    }
    with TemporaryDirectory() as td:
        path = Path(td) / "ragas_sample.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        samples, _ = load_samples_from_ragas_artifact(path)

    assert samples[0].template is None
    assert samples[0].task_type == "cross_source"


# ---------------------------------------------------------------------------
# RubricEvaluator end-to-end with fake judge
# ---------------------------------------------------------------------------


def test_evaluator_score_sample_parses_and_returns_result():
    judge = _FakeJudge([_make_payload(3, 2, 3, 2)])
    evaluator = _build_evaluator_with_fake(judge)

    sample = RubricSample(
        question="What is CVE-2023-4966?",
        answer="**Triage signal:** CRITICAL\n...",
        ground_truth="CVE-2023-4966 (Citrix Bleed), CVSS 9.4, KEV-listed.",
        sample_id="vuln_004",
        template="VulnTriage",
        task_type="vulnerability_analysis",
        difficulty="moderate",
    )
    result = evaluator.score_sample(sample)

    assert result.sample_id == "vuln_004"
    assert result.template == "VulnTriage"
    assert result.error is None
    assert result.scores.prioritization == 3
    assert result.scores.total() == 2.5
    assert "prioritization" in result.justifications
    # Judge was invoked exactly once with system + user messages.
    assert len(judge.calls) == 1
    msgs = judge.calls[0]
    assert len(msgs) == 2
    assert "CTI" in msgs[0].content  # system prompt
    assert "CVE-2023-4966" in msgs[1].content  # user prompt embeds question


def test_evaluator_score_sample_records_error_on_malformed_judge_output():
    judge = _FakeJudge(["I cannot comply."])
    evaluator = _build_evaluator_with_fake(judge)
    sample = RubricSample(
        question="Q", answer="A", sample_id="x", template="VulnTriage",
    )
    result = evaluator.score_sample(sample)
    assert result.error is not None
    assert result.scores.prioritization is None
    assert result.raw_response == "I cannot comply."


def test_evaluator_score_samples_iterates_and_preserves_order():
    payloads = [_make_payload(3, 3, 3, 3), _make_payload(0, 0, 0, 0), _make_payload(2, 2, 2, 2)]
    judge = _FakeJudge(payloads)
    # _FakeJudge only keeps one response when given a single-item list; reset:
    judge._responses = payloads
    evaluator = _build_evaluator_with_fake(judge)

    samples = [
        RubricSample(question="q1", answer="a1", sample_id="s1", template="VulnTriage"),
        RubricSample(question="q2", answer="a2", sample_id="s2", template="ThreatContext"),
        RubricSample(question="q3", answer="a3", sample_id="s3", template="CrossSourceCompare"),
    ]
    results = evaluator.score_samples(samples)
    assert [r.sample_id for r in results] == ["s1", "s2", "s3"]
    assert results[0].scores.total() == 3.0
    assert results[1].scores.total() == 0.0
    assert results[2].scores.total() == 2.0


def test_evaluator_score_sample_survives_judge_exception():
    class _Boom:
        def invoke(self, _messages):
            raise RuntimeError("upstream timeout")

    evaluator = _build_evaluator_with_fake(_Boom())
    sample = RubricSample(question="Q", answer="A", sample_id="x")
    result = evaluator.score_sample(sample)
    assert result.error is not None
    assert "upstream timeout" in result.error
    assert result.scores.prioritization is None


# ---------------------------------------------------------------------------
# DIMENSIONS sanity
# ---------------------------------------------------------------------------


def test_dimensions_match_rubric_score_fields():
    """Guards against accidental drift between the dimension tuple and the dataclass."""
    score_fields = set(RubricScore().__dict__)
    assert set(DIMENSIONS) == score_fields
