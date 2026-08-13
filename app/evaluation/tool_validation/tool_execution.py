"""ToolExecution — vendor-neutral observed tool invocation.

WHY
---
LangSmith runs, LangGraph tool nodes, OpenAI tool_calls, and MCP
``tools/call`` payloads all look different. ``ToolExecution`` is the
**internal** execution record the validator and reports understand.
Adapters (``ToolTraceMapper``, LangGraph mappers) convert vendor
shapes → ``ToolExecution``; nothing in ``ToolValidator`` parses LangSmith
objects directly.

Java SDET analogy: a normalized ``HttpCallRecord`` after your test
harness finishes talking to WireMock / browser DevTools — not the raw
Chrome CDP JSON.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolExecutionStatus(str, Enum):
    """Canonical tool execution outcomes.

    Using an ``Enum`` (instead of free-form strings) prevents typos,
    enables exhaustiveness checks, and keeps reports / LangSmith metadata
    aligned across LangGraph, future MCP, and OpenAI Agents adapters.
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    RETRY = "retry"
    CANCELLED = "cancelled"

    @classmethod
    def coerce(cls, value: Any) -> ToolExecutionStatus:
        """Normalize vendor / legacy status strings into this enum.

        Args:
            value: Raw status (enum, str, or other).

        Returns:
            ``ToolExecutionStatus`` member.
        """
        if isinstance(value, ToolExecutionStatus):
            return value
        if value is None:
            return cls.FAILED

        normalized = str(value).strip().lower()
        aliases: dict[str, ToolExecutionStatus] = {
            "success": cls.SUCCESS,
            "ok": cls.SUCCESS,
            "pass": cls.SUCCESS,
            "passed": cls.SUCCESS,
            "completed": cls.SUCCESS,
            "failed": cls.FAILED,
            "fail": cls.FAILED,
            "failure": cls.FAILED,
            "error": cls.FAILED,  # Sprint 10.2 literal alias
            "unknown": cls.FAILED,
            "skipped": cls.SKIPPED,
            "skip": cls.SKIPPED,
            "timeout": cls.TIMEOUT,
            "timed_out": cls.TIMEOUT,
            "retry": cls.RETRY,
            "retrying": cls.RETRY,
            "cancelled": cls.CANCELLED,
            "canceled": cls.CANCELLED,
        }
        return aliases.get(normalized, cls.FAILED)


class ToolExecutionMetrics(BaseModel):
    """Operational metrics for one tool invocation (Sprint 12).

    Tracks queue time, execution time, retries, failure reason, and status
    to improve future MCP integration and quality-gate latency rules.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    queue_time_ms: float = Field(default=0.0, ge=0.0)
    execution_time_ms: float = Field(default=0.0, ge=0.0)
    retries: int = Field(default=0, ge=0)
    failure_reason: str | None = None
    status: ToolExecutionStatus = Field(default=ToolExecutionStatus.FAILED)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @classmethod
    def from_latency(
        cls,
        *,
        execution_time_ms: float,
        status: ToolExecutionStatus,
        retries: int = 0,
        queue_time_ms: float = 0.0,
        failure_reason: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> ToolExecutionMetrics:
        """Build metrics from a simple latency measurement."""
        return cls(
            queue_time_ms=queue_time_ms,
            execution_time_ms=max(0.0, execution_time_ms),
            retries=retries,
            failure_reason=failure_reason,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
        )


class ToolExecution(BaseModel):
    """One observed tool invocation after adapter normalization.

    Attributes:
        tool_name: Tool / function name that ran.
        arguments: Arguments passed to the tool.
        output: Tool return value (any JSON-serializable payload).
        started_at: Optional UTC start timestamp.
        finished_at: Optional UTC finish timestamp.
        latency_ms: Wall-clock duration in milliseconds when known.
        status: Coarse execution outcome (``ToolExecutionStatus``).
        error: Error message when status is not success.
        retry_count: How many retries preceded this attempt.
        trace_metadata: Vendor trace extras (run ids, node names, …).
        order: Optional 0-based position in the observed sequence.
        metrics: Optional Sprint 12 operational metrics.
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        validate_assignment=True,
    )

    tool_name: str = Field(description="Tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments",
    )
    output: Any = Field(
        default=None,
        description="Tool output / result payload",
    )
    started_at: datetime | None = Field(
        default=None,
        description="UTC start time",
    )
    finished_at: datetime | None = Field(
        default=None,
        description="UTC finish time",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Observed latency in milliseconds",
    )
    status: ToolExecutionStatus = Field(
        default=ToolExecutionStatus.FAILED,
        description="Execution status",
    )
    error: str | None = Field(
        default=None,
        description="Error message when failed",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Retry count for this invocation",
    )
    trace_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Vendor / adapter trace metadata",
    )
    order: int = Field(
        default=0,
        ge=0,
        description="0-based index in the observed tool sequence",
    )
    metrics: ToolExecutionMetrics | None = Field(
        default=None,
        description="Optional queue/execution/retry metrics (Sprint 12)",
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank tool names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be non-empty")
        return cleaned

    @field_validator("status", mode="before")
    @classmethod
    def coerce_status(cls, value: Any) -> ToolExecutionStatus:
        """Accept enum members and legacy string aliases."""
        return ToolExecutionStatus.coerce(value)

    @property
    def success(self) -> bool:
        """Return whether this execution is considered successful."""
        return self.status == ToolExecutionStatus.SUCCESS

    def ensure_metrics(self) -> ToolExecutionMetrics:
        """Return metrics, synthesizing from latency fields when absent."""
        if self.metrics is not None:
            return self.metrics
        return ToolExecutionMetrics.from_latency(
            execution_time_ms=self.latency_ms or 0.0,
            status=self.status,
            retries=self.retry_count,
            failure_reason=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


__all__ = ["ToolExecution", "ToolExecutionStatus", "ToolExecutionMetrics"]
