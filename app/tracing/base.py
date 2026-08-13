"""Tracing abstractions — Strategy interface for execution traces.

WHY traces ≠ metrics
--------------------
Metrics answer "how good?" (scores). Traces answer "what happened?"
(ordered spans: retrieve → prompt → generate → evaluate). You debug
latency and wrong tool args from traces; you gate releases on metrics.

Production analogy (Java SDET)
------------------------------
Metrics ≈ JUnit pass/fail + coverage %.
Traces ≈ distributed tracing (OpenTelemetry spans) for one request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TraceSpanData(BaseModel):
    """Payload recorded for one RAG / evaluation execution.

    Attributes:
        question: User question.
        retrieved_chunks: Retrieved evidence texts (or summaries).
        prompt: Prompt payload (system/user or concatenated).
        model: LLM model id when known.
        tokens: Token usage dictionary.
        latency_ms: End-to-end or stage latency.
        answer: Generated answer.
        evaluation_results: Metric / tool validation summaries.
        metadata: Free-form extras (correlation id, suite name, …).
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Question under execution")
    retrieved_chunks: list[str] = Field(
        default_factory=list,
        description="Retrieved chunk texts",
    )
    prompt: dict[str, Any] | str | None = Field(
        default=None,
        description="Prompt content sent to the model",
    )
    model: str | None = Field(default=None, description="Model identifier")
    tokens: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage fields",
    )
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Observed latency in milliseconds",
    )
    answer: str | None = Field(default=None, description="Generated answer")
    evaluation_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation / tool validation summary",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional span metadata",
    )


class TraceRecord(BaseModel):
    """Identifiers returned after a trace is recorded.

    Attributes:
        run_id: Vendor run identifier.
        trace_id: Vendor trace identifier (may equal run_id).
        run_url: Deep-link into the tracing UI when available.
        project: Project / workspace name.
        recorded_at: UTC timestamp when the record was finalized.
        enabled: Whether a real backend accepted the span.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = Field(default=None, description="Vendor run id")
    trace_id: str | None = Field(default=None, description="Vendor trace id")
    run_url: str | None = Field(default=None, description="UI deep link")
    project: str | None = Field(default=None, description="Tracing project name")
    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC finalize time",
    )
    enabled: bool = Field(
        default=False,
        description="True when a backend persisted the span",
    )

    @field_validator("run_id", "trace_id", "run_url", "project")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """Normalize blank strings to ``None``."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class Tracer(ABC):
    """Strategy interface for recording execution traces.

    Implementations: ``LangSmithTracer``, ``NoOpTracer``, future Phoenix, etc.
    """

    @abstractmethod
    def record(self, span: TraceSpanData) -> TraceRecord:
        """Persist a completed execution span and return identifiers.

        Args:
            span: Structured execution payload.

        Returns:
            ``TraceRecord`` with ids / URL (or disabled no-op record).
        """


class NoOpTracer(Tracer):
    """Tracer that records nothing — used when tracing is disabled.

    Still returns synthetic local ids so reports remain schema-stable
    and tests do not need special-casing for ``None`` backends.
    """

    def record(self, span: TraceSpanData) -> TraceRecord:
        """Return a disabled local trace record without calling a vendor."""
        local_id = f"local-{uuid4()}"
        return TraceRecord(
            run_id=local_id,
            trace_id=local_id,
            run_url=None,
            project=None,
            enabled=False,
        )


__all__ = [
    "TraceSpanData",
    "TraceRecord",
    "Tracer",
    "NoOpTracer",
]
