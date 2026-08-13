"""Evaluation timeline — ordered stage events for continuous evaluation.

Request → Planning → Tool Execution → LLM → Evaluation → Quality Gate → Decision

Stored on ``EvaluationContext`` so quality reports and future dashboards can
reconstruct *when* each stage happened without parsing logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TimelineStage(str, Enum):
    """Canonical continuous-evaluation pipeline stages."""

    REQUEST = "request"
    PLANNING = "planning"
    TOOL_EXECUTION = "tool_execution"
    LLM = "llm"
    EVALUATION = "evaluation"
    QUALITY_GATE = "quality_gate"
    DECISION = "decision"


class TimelineEvent(BaseModel):
    """One stage marker on the evaluation timeline."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    stage: TimelineStage
    status: str = Field(default="completed", description="started | completed | failed")
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    detail: str = Field(default="", description="Human-readable stage detail")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationTimeline(BaseModel):
    """Ordered timeline of stages for one evaluation / agent turn."""

    model_config = ConfigDict(extra="forbid")

    events: list[TimelineEvent] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def add(
        self,
        stage: TimelineStage,
        *,
        status: str = "completed",
        duration_ms: float | None = None,
        detail: str = "",
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> EvaluationTimeline:
        """Append a stage event and return self for chaining."""
        now = datetime.now(timezone.utc)
        self.events.append(
            TimelineEvent(
                stage=stage,
                status=status,
                started_at=started_at,
                finished_at=finished_at or now,
                duration_ms=duration_ms,
                detail=detail,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def stage_names(self) -> list[str]:
        """Return ordered stage name strings."""
        return [e.stage.value for e in self.events]


__all__ = ["TimelineStage", "TimelineEvent", "EvaluationTimeline"]
