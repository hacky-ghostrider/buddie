"""Result schemas for Buddie evaluation suite output (JSON + console).

EXPECTED / ANNOTATED fields stay on the golden case.
ACTUAL / RUNTIME scores live here (DeepEval + retrieval + agent checks).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

CaseOverallStatus = Literal["passed", "failed", "error", "rate_limited"]
MetricOutcome = Literal["pass", "failed", "skipped", "error", "rate_limited"]


class MetricScoreResult(BaseModel):
    """One metric score for one golden case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    score: float | None = None
    passed: bool | None = None
    threshold: float
    reason: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None
    rate_limited: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outcome(self) -> MetricOutcome:
        """Normalized metric status for JSON / Allure reporting."""
        if self.skipped:
            return "skipped"
        if self.rate_limited:
            return "rate_limited"
        if self.error:
            return "error"
        if self.passed is True:
            return "pass"
        if self.passed is False:
            return "failed"
        return "error"

    def to_public_dict(self) -> dict[str, Any]:
        """Compact dict for reports."""
        return self.model_dump(exclude_none=True)


def _score_value(metric: MetricScoreResult | None) -> float | None:
    if metric is None or metric.skipped:
        return None
    return metric.score


class CaseEvaluationResult(BaseModel):
    """Per-case evaluation result (Sprint 19 unified schema).

    Nested ``MetricScoreResult`` objects retain DeepEval detail.
    Flat nullable floats mirror the interview/result contract and are
    ``None`` when a metric is not applicable.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str | None = None
    query: str
    expected_behavior: str | None = None

    # DeepEval generation / contextual (detailed)
    faithfulness: MetricScoreResult
    answer_relevancy: MetricScoreResult
    hallucination: MetricScoreResult | None = None
    contextual_precision: MetricScoreResult
    contextual_recall: MetricScoreResult
    contextual_relevancy: MetricScoreResult | None = None
    final_response_correctness: MetricScoreResult

    # Deterministic retrieval (expected_context vs runtime retrieval_context)
    precision_at_1: float | None = None
    precision_at_3: float | None = None
    precision_at_5: float | None = None
    recall_at_1: float | None = None
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_5: float | None = None
    mrr: float | None = None

    # Deterministic agent / functional
    tool_correctness: float | None = None
    argument_correctness: float | None = None
    hitl_correctness: float | None = None
    task_completion: float | None = None

    overall_status: CaseOverallStatus
    failure_reasons: list[str] = Field(default_factory=list)
    infrastructure_error: str | None = None
    retrieval_context_count: int = 0
    expected_output: str | None = None
    actual_output: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)

    def metric_map(self) -> dict[str, MetricScoreResult]:
        """Name → DeepEval-style score objects for aggregation."""
        mapping: dict[str, MetricScoreResult] = {
            self.faithfulness.name: self.faithfulness,
            self.answer_relevancy.name: self.answer_relevancy,
            self.contextual_precision.name: self.contextual_precision,
            self.contextual_recall.name: self.contextual_recall,
            self.final_response_correctness.name: self.final_response_correctness,
        }
        if self.hallucination is not None:
            mapping[self.hallucination.name] = self.hallucination
        if self.contextual_relevancy is not None:
            mapping[self.contextual_relevancy.name] = self.contextual_relevancy
        return mapping

    def to_flat_metric_dict(self) -> dict[str, float | None]:
        """Sprint 19 flat metric contract (null when N/A)."""
        return {
            "faithfulness": _score_value(self.faithfulness),
            "answer_relevancy": _score_value(self.answer_relevancy),
            "hallucination": _score_value(self.hallucination),
            "contextual_precision": _score_value(self.contextual_precision),
            "contextual_recall": _score_value(self.contextual_recall),
            "contextual_relevancy": _score_value(self.contextual_relevancy),
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
            "final_response_correctness": _score_value(
                self.final_response_correctness
            ),
            "tool_correctness": self.tool_correctness,
            "argument_correctness": self.argument_correctness,
            "hitl_correctness": self.hitl_correctness,
            "task_completion": self.task_completion,
        }


class SuiteEvaluationReport(BaseModel):
    """Full-suite evaluation report for 28 Buddie golden cases."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    passed: int
    failed: int
    errors: int
    rate_limited: int = 0
    metric_averages: dict[str, float] = Field(default_factory=dict)
    failed_case_ids: list[str] = Field(default_factory=list)
    error_case_ids: list[str] = Field(default_factory=list)
    rate_limited_case_ids: list[str] = Field(default_factory=list)
    failure_reasons_by_case: dict[str, list[str]] = Field(default_factory=dict)
    cases: list[CaseEvaluationResult] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    annotation_summary: dict[str, Any] | None = None
    notes: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Machine-readable report payload."""
        return self.model_dump(mode="json")


__all__ = [
    "CaseEvaluationResult",
    "CaseOverallStatus",
    "MetricOutcome",
    "MetricScoreResult",
    "SuiteEvaluationReport",
]
