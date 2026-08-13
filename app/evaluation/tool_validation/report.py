"""Tool validation report contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.tool_validation.models import (
    ActualToolCall,
    ToolCallExpectation,
    ToolMatchResult,
)


class ToolValidationReport(BaseModel):
    """Aggregated tool-validation outcome for one evaluation example.

    Attributes:
        expected_tools: Flattened expected tool names.
        actual_tools: Flattened actual tool names (call order).
        expectations: Raw expectations evaluated.
        actual_calls: Raw actual calls observed.
        matches: Per-expectation results.
        passed: Overall pass (all expectations passed + no unexpected
            calls when ``allow_extra_calls`` is False).
        failures: Aggregated failure messages.
        execution_count_expected: Sum of min counts (informational).
        execution_count_actual: Number of actual calls.
        latency_ms_total: Sum of known actual latencies.
        validated_at: UTC timestamp.
        metadata: Optional extras.
    """

    model_config = ConfigDict(extra="forbid")

    expected_tools: list[str] = Field(default_factory=list)
    actual_tools: list[str] = Field(default_factory=list)
    expectations: list[ToolCallExpectation] = Field(default_factory=list)
    actual_calls: list[ActualToolCall] = Field(default_factory=list)
    matches: list[ToolMatchResult] = Field(default_factory=list)
    passed: bool = Field(description="Overall tool validation pass/fail")
    failures: list[str] = Field(default_factory=list)
    execution_count_expected: int = Field(default=0, ge=0)
    execution_count_actual: int = Field(default=0, ge=0)
    latency_ms_total: float = Field(default=0.0, ge=0.0)
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact dict suitable for traces / parent reports."""
        return {
            "passed": self.passed,
            "expected_tools": list(self.expected_tools),
            "actual_tools": list(self.actual_tools),
            "failures": list(self.failures),
            "execution_count_expected": self.execution_count_expected,
            "execution_count_actual": self.execution_count_actual,
            "latency_ms_total": self.latency_ms_total,
        }


__all__ = ["ToolValidationReport"]
