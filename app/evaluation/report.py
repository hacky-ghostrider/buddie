"""Structured evaluation report contract.

Reports are structured (not free-text logs) so results can be compared across
runs, serialized to JSON/CSV/HTML, aggregated in CI, and linked to traces
without re-parsing prose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evaluation.models import MetricResult
from app.evaluation.tool_validation.report import ToolValidationReport
from app.retrieval.models import RetrievedDocument


class EvaluationReport(BaseModel):
    """Aggregated outcome of evaluating one RAG (and optional tool) response.

    Attributes:
        question: Evaluated question.
        expected_answer: Optional golden answer.
        answer: Generated answer that was scored.
        retrieved_documents: Evidence available to the generator.
        metrics: Per-metric results in execution order.
        overall_score: Aggregate score in ``[0, 1]`` (mean of metric scores).
        passed: Whether ``overall_score`` meets the configured pass threshold
            and optional tool validation passed.
        evaluation_time: UTC timestamp when the report was finalized.
        latency: Total evaluation wall-clock latency in milliseconds.
        rag_latency_ms: Optional end-to-end RAG latency from the pipeline.
        pass_threshold: Threshold used for the ``passed`` decision.
        token_usage: Optional token accounting from generation.
        estimated_cost_usd: Optional estimated generation cost.
        langsmith_run_id: LangSmith run id when tracing is enabled.
        langsmith_trace_id: LangSmith trace id when tracing is enabled.
        langsmith_run_url: Deep link to the LangSmith UI run.
        tool_validation: Optional tool-validation report.
        metadata: Optional run-level extras (correlation id, suite name, …).
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Evaluated question")
    expected_answer: str | None = Field(
        default=None,
        description="Optional golden / expected answer",
    )
    answer: str = Field(description="Generated answer under evaluation")
    retrieved_documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Retrieved documents associated with the answer",
    )
    metrics: list[MetricResult] = Field(
        default_factory=list,
        description="Individual metric results",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Mean of metric scores in [0, 1]",
    )
    passed: bool = Field(description="Whether the evaluation passed")
    evaluation_time: datetime = Field(
        description="UTC timestamp when evaluation completed",
    )
    latency: float = Field(
        ge=0.0,
        description="Total evaluation latency in milliseconds",
    )
    rag_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional RAG pipeline latency in milliseconds",
    )
    pass_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Pass threshold applied to overall_score",
    )
    token_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional token usage from generation",
    )
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional estimated cost in USD",
    )
    langsmith_run_id: str | None = Field(
        default=None,
        description="LangSmith run id",
    )
    langsmith_trace_id: str | None = Field(
        default=None,
        description="LangSmith trace id",
    )
    langsmith_run_url: str | None = Field(
        default=None,
        description="LangSmith run URL",
    )
    tool_validation: ToolValidationReport | None = Field(
        default=None,
        description="Optional tool validation report",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional report-level metadata",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject blank questions on reports."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("EvaluationReport.question must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def passed_must_match_threshold_and_tools(self) -> EvaluationReport:
        """Ensure ``passed`` matches score threshold and tool validation."""
        score_ok = self.overall_score >= self.pass_threshold
        tools_ok = (
            True
            if self.tool_validation is None
            else self.tool_validation.passed
        )
        expected = score_ok and tools_ok
        if self.passed != expected:
            raise ValueError(
                f"passed={self.passed} is inconsistent with "
                f"overall_score={self.overall_score}, "
                f"pass_threshold={self.pass_threshold}, "
                f"tool_validation_passed={tools_ok}"
            )
        return self

    @classmethod
    def build(
        cls,
        *,
        question: str,
        answer: str,
        retrieved_documents: list[RetrievedDocument],
        metrics: list[MetricResult],
        latency_ms: float,
        pass_threshold: float,
        metadata: dict[str, Any] | None = None,
        evaluation_time: datetime | None = None,
        expected_answer: str | None = None,
        rag_latency_ms: float | None = None,
        token_usage: dict[str, Any] | None = None,
        estimated_cost_usd: float | None = None,
        langsmith_run_id: str | None = None,
        langsmith_trace_id: str | None = None,
        langsmith_run_url: str | None = None,
        tool_validation: ToolValidationReport | None = None,
    ) -> EvaluationReport:
        """Construct a validated report from metric results.

        Args:
            question: Evaluated question.
            answer: Generated answer.
            retrieved_documents: Retrieval evidence.
            metrics: Completed metric results.
            latency_ms: Total evaluation duration.
            pass_threshold: Threshold for score-based pass.
            metadata: Optional extras.
            evaluation_time: Optional override; defaults to UTC now.
            expected_answer: Optional golden answer.
            rag_latency_ms: Optional RAG latency.
            token_usage: Optional token dict.
            estimated_cost_usd: Optional cost estimate.
            langsmith_run_id: Optional LangSmith run id.
            langsmith_trace_id: Optional LangSmith trace id.
            langsmith_run_url: Optional LangSmith URL.
            tool_validation: Optional tool validation report.

        Returns:
            Validated ``EvaluationReport``.

        Raises:
            ValueError: When fields fail Pydantic validation.
        """
        if metrics:
            overall = sum(item.score for item in metrics) / len(metrics)
        else:
            overall = 0.0

        score_ok = overall >= pass_threshold
        tools_ok = True if tool_validation is None else tool_validation.passed

        return cls(
            question=question,
            expected_answer=expected_answer,
            answer=answer,
            retrieved_documents=list(retrieved_documents),
            metrics=list(metrics),
            overall_score=round(overall, 6),
            passed=score_ok and tools_ok,
            evaluation_time=evaluation_time
            or datetime.now(timezone.utc),
            latency=max(0.0, latency_ms),
            rag_latency_ms=rag_latency_ms,
            pass_threshold=pass_threshold,
            token_usage=dict(token_usage or {}),
            estimated_cost_usd=estimated_cost_usd,
            langsmith_run_id=langsmith_run_id,
            langsmith_trace_id=langsmith_trace_id,
            langsmith_run_url=langsmith_run_url,
            tool_validation=tool_validation,
            metadata=dict(metadata or {}),
        )
