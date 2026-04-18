"""
Structured Rubric LLM-as-Judge Evaluator (Phase 2, spec section 8.4).

Scores rendered RAG answers on four 0-3 dimensions:

  - Prioritization  (severity signal visibility and rationale)
  - Actionability   (discrete, concrete, grounded mitigations)
  - Completeness    (template fields present or gap-acknowledged)
  - Traceability    (every factual claim cited)

Unlike the Field-Coverage evaluator, which works deterministically on
the Fact Bundle, this evaluator asks an LLM to judge the rendered
analyst-facing output as a whole. It complements RAGAS (grounding
fidelity) and Field-Coverage (analyst utility on known fields) by
capturing analyst-utility signals that do not reduce to fixed fields —
e.g. "is severity prominent at the top", "are mitigations concrete".

Two execution modes:

  1. **Offline rescoring.** Load an existing ``ragas_*.json`` artifact and
     rescore its per-sample records. No pipeline run needed; the RAGAS
     artifacts already contain ``question``, ``answer``, ``ground_truth``,
     ``template``, ``task_type``, and ``difficulty``. This is the
     recommended mode for comparing Legacy vs Phase-2 output under the
     same judge without re-running the generator.

  2. **Live scoring.** Pass a list of ``RubricSample`` records (or
     ``RAGResponse`` + metadata) directly. Used by the smoke runner and
     integrated pipeline+rubric runs.

Artifact schema
---------------
``data/evaluation_results/setup_<x>/rubric_<experiment>_<ts>.json``::

    {
      "experiment_name": "...",
      "judge_model": "openrouter:openai/gpt-4o-mini",
      "source_artifact": "ragas_hybrid_templated_phase2_raw_28q_*.json" | null,
      "timestamp": "2026-04-15T08:12:33",
      "num_samples": 28,
      "dimensions": ["prioritization", "actionability", "completeness",
                     "traceability"],
      "aggregated": {
        "prioritization_mean": 2.1, "actionability_mean": 1.7,
        "completeness_mean": 2.4, "traceability_mean": 1.3,
        "total_mean": 1.875
      },
      "per_template": {"VulnTriage": {...}, ...},
      "per_task_type": {"vulnerability_analysis": {...}, ...},
      "per_sample": [{"sample_id": "vuln_004", "scores": {...},
                      "justifications": {...}, "raw_response": "..."},
                     ...]
    }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..utils.config import (
    get_project_root,
    load_config,
    suppress_noisy_third_party_logs,
)
from ._judge_llm import build_judge_llm

logger = logging.getLogger(__name__)
suppress_noisy_third_party_logs()


DIMENSIONS = ("prioritization", "actionability", "completeness", "traceability")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RubricSample:
    """One input row for the rubric judge."""

    question: str
    answer: str
    ground_truth: str = ""
    sample_id: str | None = None
    template: str | None = None
    task_type: str | None = None
    difficulty: str | None = None


@dataclass
class RubricScore:
    """Per-dimension score + optional justification from the judge."""

    prioritization: int | None = None
    actionability: int | None = None
    completeness: int | None = None
    traceability: int | None = None

    def total(self) -> float | None:
        parts = [
            self.prioritization,
            self.actionability,
            self.completeness,
            self.traceability,
        ]
        if any(p is None for p in parts):
            return None
        return sum(parts) / 4.0


@dataclass
class RubricSampleResult:
    sample_id: str | None
    template: str | None
    task_type: str | None
    difficulty: str | None
    scores: RubricScore
    justifications: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "template": self.template,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "scores": asdict(self.scores),
            "total": self.scores.total(),
            "justifications": dict(self.justifications),
            "raw_response": self.raw_response,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_RUBRIC_SYSTEM_PROMPT = """You are an expert Cyber Threat Intelligence (CTI) analyst and evaluator.

You review the output of a RAG system that produces triage cards for CTI analysts. Your job is to score a single analyst-facing answer on four analytic-quality dimensions, each on a 0-3 integer scale.

Dimensions and rubric:

1. Prioritization (severity signal visibility)
   0 — No severity signal, or severity is misleading / contradicts the ground truth.
   1 — Severity is mentioned but buried in the text; not actionable at a glance.
   2 — Severity is clearly visible, with a rationale (e.g. CVSS, KEV listing).
   3 — Severity is prominent at the top of the answer AND has a clear rationale tying it to concrete signals (CVSS score, KEV listing, ransomware use, exploitation status).

2. Actionability (defensive guidance quality)
   0 — No mitigations offered, or only generic "apply patches" / "follow best practices".
   1 — Generic mitigations only; nothing product-specific or actionable.
   2 — Some concrete items present, some generic; at least one points to a specific control, patch, or configuration.
   3 — Discrete, specific, source-grounded mitigations an analyst could hand to an operator (e.g. specific KB number, specific config change, specific IOC to block).

4. Completeness (template-field coverage)
   0 — Major fields missing and NOT acknowledged as gaps (silent omission).
   1 — Many fields missing; the answer reads as incomplete.
   2 — Most template fields present; minor gaps.
   3 — All expected fields present OR any missing fields are explicitly flagged as an information gap.

4. Traceability (citation quality)
   0 — No citations, or citations that do not correspond to any retrieved source.
   1 — Sparse citations; many factual claims unattributed.
   2 — Most factual claims cited with a visible source marker.
   3 — Every factual claim has a citation and the citations look correct (source label + doc id).

Important guidance:
- Judge only what is present in the answer. Do not penalize the system for gaps the answer explicitly acknowledges.
- Treat citation markers of the form [Source: <label>] or [Source: <label> | <doc_id>] as valid citations. Raw URLs or section-name tokens like [initial_access] do NOT count as valid citations.
- Use the ground truth only as a reality check for Prioritization and Completeness, not to penalize stylistic differences.
- Scores are integers in {0, 1, 2, 3}. Do not use fractional values.

Output format: respond with a single JSON object and nothing else. The JSON object must contain exactly these keys:
  "prioritization":  {"score": int, "justification": "<one sentence>"},
  "actionability":   {"score": int, "justification": "<one sentence>"},
  "completeness":    {"score": int, "justification": "<one sentence>"},
  "traceability":    {"score": int, "justification": "<one sentence>"}
"""


def _build_user_prompt(sample: RubricSample) -> str:
    template_line = (
        f"Routed template: {sample.template}\n" if sample.template else ""
    )
    gt_block = sample.ground_truth.strip() or "(no ground truth provided)"
    return (
        f"{template_line}"
        f"Analyst question:\n{sample.question.strip()}\n\n"
        f"Ground truth (for reality-check only):\n{gt_block}\n\n"
        f"Analyst answer to score:\n---\n{sample.answer.strip()}\n---\n\n"
        "Return the JSON object now."
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_blob(text: str) -> str | None:
    """Best-effort extract the first top-level JSON object from a string.

    The judge should return JSON-only (enforced via response_format when
    supported), but some providers still wrap the payload in prose or
    markdown code fences. We try strict-parse first, then fall back to
    the largest ``{...}`` span.
    """
    text = text.strip()
    if not text:
        return None
    # Fast path: entire response is JSON.
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    # Fallback: grab the first balanced-looking {...} block. This is
    # imperfect on deeply nested content, but sufficient for our flat
    # 4-key schema.
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            return None
    return None


def _coerce_score(value: Any) -> int | None:
    """Clamp a judge-supplied score to the 0..3 integer range."""
    try:
        iv = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if 0 <= iv <= 3:
        return iv
    # Out-of-range is almost always a judge mistake; clamp rather than drop.
    return max(0, min(3, iv))


def _parse_judge_payload(raw: str) -> tuple[RubricScore, dict[str, str], str | None]:
    """Parse the judge's JSON response into a :class:`RubricScore`.

    Returns ``(scores, justifications, error)``. ``error`` is ``None`` on
    success; on failure, ``scores`` will contain ``None`` fields and
    ``error`` carries the reason.
    """
    blob = _extract_json_blob(raw)
    if blob is None:
        return RubricScore(), {}, "judge response was not valid JSON"

    try:
        data = json.loads(blob)
    except Exception as exc:  # pragma: no cover — _extract already validated
        return RubricScore(), {}, f"json parse failure: {exc}"

    if not isinstance(data, dict):
        return RubricScore(), {}, "judge response JSON was not an object"

    scores = RubricScore()
    justifications: dict[str, str] = {}
    missing: list[str] = []
    for dim in DIMENSIONS:
        entry = data.get(dim)
        if isinstance(entry, dict):
            score = _coerce_score(entry.get("score"))
            justification = str(entry.get("justification", "")).strip()
        else:
            # Some judges return {dim: <int>} flat instead of nested.
            score = _coerce_score(entry) if entry is not None else None
            justification = ""
        if score is None:
            missing.append(dim)
        setattr(scores, dim, score)
        if justification:
            justifications[dim] = justification

    error = f"missing or invalid dimensions: {missing}" if missing else None
    return scores, justifications, error


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class RubricEvaluator:
    """LLM-as-Judge evaluator for the Structured Rubric (spec section 8.4)."""

    def __init__(self, provider: str | None = None):
        config = load_config()
        eval_config = config["evaluation"]["ragas"]
        llm_config = config["llm"]
        self._provider = (
            provider or eval_config.get("eval_llm_provider", "ollama")
        ).lower()

        self.judge_model_label, raw_llm = build_judge_llm(
            provider=self._provider,
            eval_config=eval_config,
            ollama_base_url=llm_config["base_url"],
            timeout=180,
        )

        # Request structured JSON output where the provider supports it.
        # OpenAI and OpenRouter (via OpenAI-compatible API) accept
        # ``response_format={"type":"json_object"}``. Ollama's Chat API
        # ignores unknown kwargs silently, so binding is safe there too.
        try:
            self.judge_llm = raw_llm.bind(response_format={"type": "json_object"})
        except Exception:  # pragma: no cover — defensive
            self.judge_llm = raw_llm

        self.results_dir = get_project_root() / config["evaluation"]["results_dir"]
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("RubricEvaluator initialized (judge=%s)", self.judge_model_label)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_sample(self, sample: RubricSample) -> RubricSampleResult:
        """Score a single sample, returning a :class:`RubricSampleResult`."""
        messages = [
            SystemMessage(content=_RUBRIC_SYSTEM_PROMPT),
            HumanMessage(content=_build_user_prompt(sample)),
        ]
        try:
            response = self.judge_llm.invoke(messages)
            raw = getattr(response, "content", "") or ""
        except Exception as exc:
            logger.warning(
                "Rubric judge call failed for sample_id=%s: %s", sample.sample_id, exc
            )
            return RubricSampleResult(
                sample_id=sample.sample_id,
                template=sample.template,
                task_type=sample.task_type,
                difficulty=sample.difficulty,
                scores=RubricScore(),
                raw_response="",
                error=f"judge invocation failed: {exc}",
            )

        scores, justifications, parse_error = _parse_judge_payload(raw)
        if parse_error:
            logger.warning(
                "Rubric parse issue for sample_id=%s: %s", sample.sample_id, parse_error
            )

        return RubricSampleResult(
            sample_id=sample.sample_id,
            template=sample.template,
            task_type=sample.task_type,
            difficulty=sample.difficulty,
            scores=scores,
            justifications=justifications,
            raw_response=raw,
            error=parse_error,
        )

    def score_samples(self, samples: list[RubricSample]) -> list[RubricSampleResult]:
        """Score a list of samples sequentially."""
        results: list[RubricSampleResult] = []
        for index, sample in enumerate(samples, start=1):
            logger.info(
                "Rubric scoring %d/%d (sample_id=%s)",
                index,
                len(samples),
                sample.sample_id,
            )
            results.append(self.score_sample(sample))
        return results

    # ------------------------------------------------------------------
    # Artifact writing
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        results: list[RubricSampleResult],
        experiment_name: str,
        source_artifact: str | None = None,
        run_metadata: dict | None = None,
    ) -> dict:
        """Aggregate results and persist a trace-rich JSON artifact."""
        aggregated = aggregate_rubric_scores(results)
        artifact = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "judge_model": self.judge_model_label,
            "source_artifact": source_artifact,
            "num_samples": len(results),
            "dimensions": list(DIMENSIONS),
            "aggregated": aggregated["overall"],
            "per_template": aggregated["per_template"],
            "per_task_type": aggregated["per_task_type"],
            "per_difficulty": aggregated["per_difficulty"],
            "run_metadata": run_metadata or {},
            "per_sample": [result.to_dict() for result in results],
        }

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"rubric_{experiment_name}_{timestamp_str}.json"
        artifact["output_path"] = str(output_path)

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, default=str)

        logger.info("Rubric artifact saved: %s", output_path)
        return artifact


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _group_means(
    results: list[RubricSampleResult], key: str
) -> dict[str, dict[str, float | int]]:
    """Aggregate mean per dimension grouped by an attribute of the result."""
    buckets: dict[str, list[RubricSampleResult]] = {}
    for result in results:
        group = getattr(result, key, None)
        if not group:
            continue
        buckets.setdefault(group, []).append(result)
    return {group: _overall_means(rows) for group, rows in sorted(buckets.items())}


def _overall_means(results: list[RubricSampleResult]) -> dict[str, float | int]:
    """Compute dimension + total means across a list of results."""
    summary: dict[str, float | int] = {"n": len(results)}
    for dim in DIMENSIONS:
        values = [
            getattr(result.scores, dim)
            for result in results
            if getattr(result.scores, dim) is not None
        ]
        summary[f"{dim}_mean"] = round(mean(values), 4) if values else None
    totals = [result.scores.total() for result in results if result.scores.total() is not None]
    summary["total_mean"] = round(mean(totals), 4) if totals else None
    summary["errors"] = sum(1 for result in results if result.error)
    return summary


def aggregate_rubric_scores(results: list[RubricSampleResult]) -> dict:
    """Aggregate rubric results overall and by template / task_type / difficulty."""
    return {
        "overall": _overall_means(results),
        "per_template": _group_means(results, "template"),
        "per_task_type": _group_means(results, "task_type"),
        "per_difficulty": _group_means(results, "difficulty"),
    }


# ---------------------------------------------------------------------------
# Offline loader
# ---------------------------------------------------------------------------


def load_samples_from_ragas_artifact(path: str | Path) -> tuple[list[RubricSample], dict]:
    """Load per-sample records from a RAGAS artifact for offline rescoring.

    Returns ``(samples, source_meta)`` where ``source_meta`` is a small
    dict with the artifact's ``experiment_name``, ``retrieval_mode``,
    and ``num_samples`` for traceability in the rubric artifact.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    per_sample = data.get("per_sample") or []
    samples: list[RubricSample] = []
    for record in per_sample:
        sample = RubricSample(
            question=record.get("question", ""),
            answer=record.get("answer", ""),
            ground_truth=record.get("ground_truth", ""),
            sample_id=record.get("sample_id") or record.get("id"),
            # Template is present on templated runs; legacy runs leave it blank.
            template=record.get("template"),
            task_type=record.get("task_type"),
            difficulty=record.get("difficulty"),
        )
        if not sample.question or not sample.answer:
            logger.warning(
                "Skipping record with missing question/answer: %s", sample.sample_id
            )
            continue
        samples.append(sample)

    source_meta = {
        "experiment_name": data.get("experiment_name"),
        "retrieval_mode": data.get("retrieval_mode"),
        "num_samples": data.get("num_samples"),
        "source_path": str(path),
    }
    return samples, source_meta
