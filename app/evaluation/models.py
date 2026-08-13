"""Shared evaluation contracts (inputs, metric results, golden examples).

These models are tool-independent: DeepEval / RAGAS adapters map *into*
these shapes, not replace them.

``EvaluationContext`` lives in ``app.evaluation.context`` (Sprint 10.2) and
is re-exported here for backward compatibility with Sprint 9 imports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.context import EvaluationContext

DifficultyLevel = Literal["easy", "medium", "hard"]

# Re-export so ``from app.evaluation.models import EvaluationContext`` keeps working.
__all_context__ = ["EvaluationContext"]


class MetricResult(BaseModel):
    """Outcome of a single metric execution.

    Attributes:
        name: Metric identifier (matches ``Metric.name()``).
        score: Numeric score in ``[0, 1]`` (higher is better).
        passed: Whether this metric individually passed its threshold.
        details: Structured extras for debugging / dashboards later.
        error: Populated when the metric failed or timed out.
        latency_ms: Wall-clock duration of this metric run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Metric name")
    score: float = Field(ge=0.0, le=1.0, description="Score in [0, 1]")
    passed: bool = Field(description="Whether the metric passed")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific detail payload",
    )
    error: str | None = Field(
        default=None,
        description="Error message when the metric failed",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Metric execution latency in milliseconds",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank metric names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("MetricResult.name must be non-empty")
        return cleaned


class GoldenExample(BaseModel):
    """One labeled example in a golden (ground-truth) evaluation dataset.

    Offline evaluation scores RAG (and later agent) outputs against known-good
    answers, sources, and tool contracts — like a curated regression suite
    in QA, not a live production log.

    Attributes:
        question: Input question.
        expected_answer: Canonical / acceptable reference answer.
        expected_sources: Source ids or paths that should be retrieved.
        expected_tools: Tool names the agent is expected to call (Sprint 11).
        expected_tool_arguments: Per-tool argument maps aligned by index.
        expected_tool_order: Ordered tool names when sequence matters.
        difficulty: Coarse difficulty bucket for stratified reporting.
        category: Suite category (e.g. ``rag``, ``retrieval``, ``agent``).
        tags: Labels for filtering suites (e.g. ``chunking``, ``policy``).
        id: Optional stable example identifier.
        metadata: Optional free-form extras.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Golden dataset question")
    expected_answer: str = Field(description="Expected / reference answer")
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Expected source identifiers",
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Expected tool names for agent evaluation",
    )
    expected_tool_arguments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Expected tool argument maps aligned with tools/order",
    )
    expected_tool_order: list[str] = Field(
        default_factory=list,
        description="Expected tool execution order",
    )
    difficulty: DifficultyLevel = Field(
        default="medium",
        description="Difficulty: easy | medium | hard",
    )
    category: str = Field(
        default="rag",
        description="Example category for suite filtering",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags for suite filtering",
    )
    id: str | None = Field(
        default=None,
        description="Optional stable golden example id",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional example metadata",
    )

    @field_validator("question", "expected_answer")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        """Reject blank question / expected answer strings."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("GoldenExample text fields must be non-empty")
        return cleaned

    @field_validator("category")
    @classmethod
    def category_must_not_be_blank(cls, value: str) -> str:
        """Reject blank categories."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("GoldenExample.category must be non-empty")
        return cleaned

    @field_validator(
        "tags",
        "expected_sources",
        "expected_tools",
        "expected_tool_order",
    )
    @classmethod
    def strip_string_list_entries(cls, value: list[str]) -> list[str]:
        """Normalize list entries by stripping whitespace and dropping blanks."""
        return [item.strip() for item in value if item and item.strip()]


__all__ = [
    "EvaluationContext",
    "MetricResult",
    "GoldenExample",
    "DifficultyLevel",
]
