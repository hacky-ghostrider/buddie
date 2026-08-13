"""Typed tool results (Sprint 12 hardening).

``ToolResult[T]`` wraps successful/failed payloads with strong typing where
practical. Metrics live on ``ToolExecution.metrics`` /
``ToolExecutionMetrics`` — this module focuses on typed *payloads*.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.tool_validation.tool_execution import ToolExecutionMetrics

T = TypeVar("T")


class ToolResult(BaseModel, Generic[T]):
    """Strongly typed tool output wrapper.

    Prefer embedding ``ToolResult`` payloads inside ``ToolExecution.output``
    rather than inventing a parallel execution path.

    Attributes:
        tool_name: Tool that produced the result.
        success: Whether the tool succeeded.
        data: Typed payload when successful.
        error: Error message when unsuccessful.
        metrics: Optional execution metrics.
        metadata: Free-form extras.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tool_name: str
    success: bool = True
    data: T | None = None
    error: str | None = None
    metrics: ToolExecutionMetrics | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        tool_name: str,
        data: T,
        *,
        metrics: ToolExecutionMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult[T]:
        """Create a successful typed result."""
        return cls(
            tool_name=tool_name,
            success=True,
            data=data,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def fail(
        cls,
        tool_name: str,
        error: str,
        *,
        metrics: ToolExecutionMetrics | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult[T]:
        """Create a failed typed result."""
        return cls(
            tool_name=tool_name,
            success=False,
            data=None,
            error=error,
            metrics=metrics,
            metadata=dict(metadata or {}),
        )


class CalculatorResultData(BaseModel):
    """Typed calculator tool payload."""

    model_config = ConfigDict(extra="forbid")

    result: float | int
    expression: str


class SearchResultData(BaseModel):
    """Typed search / search_docs payload (subset)."""

    model_config = ConfigDict(extra="forbid")

    query: str = ""
    results: list[dict[str, Any]] = Field(default_factory=list)
    answer: str | None = None


__all__ = [
    "ToolResult",
    "CalculatorResultData",
    "SearchResultData",
]
