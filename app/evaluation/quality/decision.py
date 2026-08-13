"""QualityDecision — PASS / WARNING / FAIL outcome of a gate evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.quality.models import QualityRuleResult


class QualityStatus(str, Enum):
    """Terminal quality-gate statuses."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class QualityRecommendation(BaseModel):
    """Actionable remediation tied to a failed or warned rule.

    Attributes:
        rule_id: Related quality rule id (string for serialization ease).
        category: Coarse remediation category.
        message: Human-readable recommendation.
        actions: Concrete follow-up steps.
        priority: ``high`` | ``medium`` | ``low``.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="Related rule identifier")
    category: str = Field(description="Remediation category")
    message: str = Field(description="Recommendation summary")
    actions: list[str] = Field(default_factory=list, description="Concrete steps")
    priority: str = Field(default="medium", description="high | medium | low")

    @field_validator("rule_id", "category", "message")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank recommendation fields."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("QualityRecommendation text fields must be non-empty")
        return cleaned


class QualityDecision(BaseModel):
    """Final quality-gate decision for one evaluation (or suite aggregate).

    Fields:
        status: ``PASS`` | ``WARNING`` | ``FAIL``.
        reason: Short explanation of the decision.
        failed_rules: Rules that forced FAIL.
        warnings: Rules that produced WARNING severity.
        overall_score: Score used for the decision (usually report overall).
        timestamp: UTC decision time.
        correlation_id: Request / pipeline correlation id.
        rule_results: Full per-rule outcomes.
        recommendations: Actionable remediations for failures / warnings.
        metadata: Free-form extras (suite name, run id, …).
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    status: QualityStatus = Field(description="PASS | WARNING | FAIL")
    reason: str = Field(description="Decision rationale")
    failed_rules: list[str] = Field(
        default_factory=list,
        description="Rule ids that caused FAIL",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Rule ids that caused WARNING",
    )
    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall score considered by the gate",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC decision timestamp",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation id for the evaluated request",
    )
    rule_results: list[QualityRuleResult] = Field(
        default_factory=list,
        description="Per-rule evaluation outcomes",
    )
    recommendations: list[QualityRecommendation] = Field(
        default_factory=list,
        description="Actionable remediations",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True when status is PASS (warnings still fail this check)."""
        return self.status == QualityStatus.PASS

    @property
    def is_blocking_failure(self) -> bool:
        """Return True when status is FAIL."""
        return self.status == QualityStatus.FAIL


__all__ = [
    "QualityStatus",
    "QualityRecommendation",
    "QualityDecision",
]
