"""Quality-gate rule contracts and threshold configuration.

Rules are declarative: the engine evaluates ``EvaluationReport`` (plus
optional tool / cost / latency signals) against thresholds loaded from
``Settings``. No hard-coded production thresholds live in gate logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuleOperator(str, Enum):
    """Comparison operator for a quality rule threshold."""

    MINIMUM = "minimum"
    MAXIMUM = "maximum"


class RuleSeverity(str, Enum):
    """How a failed rule contributes to the overall gate decision."""

    FAIL = "fail"
    WARNING = "warning"


class QualityRuleId(str, Enum):
    """Stable identifiers for built-in quality rules."""

    MIN_FAITHFULNESS = "min_faithfulness"
    MAX_HALLUCINATION = "max_hallucination"
    MIN_ANSWER_RELEVANCY = "min_answer_relevancy"
    MIN_CONTEXT_PRECISION = "min_context_precision"
    MIN_CONTEXT_RECALL = "min_context_recall"
    MAX_TOOL_FAILURES = "max_tool_failures"
    MAX_TOOL_LATENCY = "max_tool_latency"
    MAX_TOTAL_LATENCY = "max_total_latency"
    MAX_COST = "max_cost"
    MIN_OVERALL_SCORE = "min_overall_score"


MetricAlias = Literal[
    "faithfulness",
    "hallucination",
    "answer_relevancy",
    "relevancy",
    "contextual_precision",
    "context_precision",
    "contextual_recall",
    "context_recall",
    "overall_score",
]


class QualityRule(BaseModel):
    """One configurable quality threshold.

    Attributes:
        rule_id: Stable rule identifier.
        name: Human-readable label.
        operator: Whether the observed value must be >= or <= threshold.
        threshold: Numeric cut-off.
        metric_aliases: Metric names to resolve from ``EvaluationReport``.
        severity: ``fail`` forces FAIL; ``warning`` can yield WARNING.
        enabled: When False, the rule is skipped.
        description: Optional explanation for reports / demos.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    rule_id: QualityRuleId = Field(description="Stable rule id")
    name: str = Field(description="Display name")
    operator: RuleOperator = Field(description="minimum | maximum")
    threshold: float = Field(description="Configured threshold value")
    metric_aliases: list[str] = Field(
        default_factory=list,
        description="Metric names used to resolve the observed value",
    )
    severity: RuleSeverity = Field(
        default=RuleSeverity.FAIL,
        description="Contribution of a failure to the gate decision",
    )
    enabled: bool = Field(default=True, description="Whether the rule is active")
    description: str = Field(default="", description="Rule rationale")

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank rule names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("QualityRule.name must be non-empty")
        return cleaned


class QualityRuleResult(BaseModel):
    """Outcome of evaluating one quality rule against a report."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    rule_id: QualityRuleId
    name: str
    operator: RuleOperator
    threshold: float
    observed: float | None = Field(
        default=None,
        description="Observed value when resolvable; None if metric missing",
    )
    passed: bool
    severity: RuleSeverity
    skipped: bool = Field(
        default=False,
        description="True when the rule could not resolve an observed value",
    )
    message: str = Field(default="", description="Human-readable outcome")
    details: dict[str, Any] = Field(default_factory=dict)


class QualityGateThresholds(BaseModel):
    """Configured numeric thresholds for the quality-gate engine.

    Loaded from ``Settings`` so CI / environments can tune gates without
    code changes.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_faithfulness: float = Field(default=0.7, ge=0.0, le=1.0)
    max_hallucination: float = Field(default=0.3, ge=0.0, le=1.0)
    min_relevancy: float = Field(default=0.7, ge=0.0, le=1.0)
    min_context_precision: float = Field(default=0.6, ge=0.0, le=1.0)
    min_context_recall: float = Field(default=0.6, ge=0.0, le=1.0)
    max_tool_failures: int = Field(default=0, ge=0)
    max_tool_latency: float = Field(default=60_000.0, ge=0.0)
    max_total_latency: float = Field(default=120_000.0, ge=0.0)
    max_cost: float = Field(default=1.0, ge=0.0)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    warning_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    skip_missing_metrics: bool = Field(
        default=True,
        description="Skip metric rules when the metric is absent from the report",
    )

    @classmethod
    def from_settings(cls, settings: Any) -> QualityGateThresholds:
        """Build thresholds from application ``Settings``.

        Args:
            settings: ``Settings`` instance (duck-typed for testability).

        Returns:
            Populated ``QualityGateThresholds``.
        """
        return cls(
            enabled=bool(getattr(settings, "quality_gate_enabled", True)),
            min_faithfulness=float(getattr(settings, "min_faithfulness", 0.7)),
            max_hallucination=float(getattr(settings, "max_hallucination", 0.3)),
            min_relevancy=float(getattr(settings, "min_relevancy", 0.7)),
            min_context_precision=float(
                getattr(settings, "min_context_precision", 0.6)
            ),
            min_context_recall=float(getattr(settings, "min_context_recall", 0.6)),
            max_tool_failures=int(getattr(settings, "max_tool_failures", 0)),
            max_tool_latency=float(getattr(settings, "max_tool_latency", 60_000.0)),
            max_total_latency=float(getattr(settings, "max_total_latency", 120_000.0)),
            max_cost=float(getattr(settings, "max_cost", 1.0)),
            pass_threshold=float(
                getattr(
                    settings,
                    "quality_pass_threshold",
                    getattr(settings, "default_pass_threshold", 0.7),
                )
            ),
            warning_threshold=float(getattr(settings, "warning_threshold", 0.6)),
        )


def build_default_rules(thresholds: QualityGateThresholds) -> list[QualityRule]:
    """Construct the standard rule set from configured thresholds.

    Args:
        thresholds: Numeric thresholds from configuration.

    Returns:
        Ordered list of enabled ``QualityRule`` instances.
    """
    return [
        QualityRule(
            rule_id=QualityRuleId.MIN_FAITHFULNESS,
            name="Minimum Faithfulness",
            operator=RuleOperator.MINIMUM,
            threshold=thresholds.min_faithfulness,
            metric_aliases=["faithfulness"],
            severity=RuleSeverity.FAIL,
            description="Answer must be grounded in retrieved context",
        ),
        QualityRule(
            rule_id=QualityRuleId.MAX_HALLUCINATION,
            name="Maximum Hallucination",
            operator=RuleOperator.MAXIMUM,
            threshold=thresholds.max_hallucination,
            metric_aliases=["hallucination"],
            severity=RuleSeverity.FAIL,
            description="Hallucinated content must stay below the ceiling",
        ),
        QualityRule(
            rule_id=QualityRuleId.MIN_ANSWER_RELEVANCY,
            name="Minimum Answer Relevancy",
            operator=RuleOperator.MINIMUM,
            threshold=thresholds.min_relevancy,
            metric_aliases=["answer_relevancy", "relevancy"],
            severity=RuleSeverity.FAIL,
            description="Answer must address the user question",
        ),
        QualityRule(
            rule_id=QualityRuleId.MIN_CONTEXT_PRECISION,
            name="Minimum Context Precision",
            operator=RuleOperator.MINIMUM,
            threshold=thresholds.min_context_precision,
            metric_aliases=["contextual_precision", "context_precision"],
            severity=RuleSeverity.WARNING,
            description="Retrieved context should be precise",
        ),
        QualityRule(
            rule_id=QualityRuleId.MIN_CONTEXT_RECALL,
            name="Minimum Context Recall",
            operator=RuleOperator.MINIMUM,
            threshold=thresholds.min_context_recall,
            metric_aliases=["contextual_recall", "context_recall"],
            severity=RuleSeverity.WARNING,
            description="Retrieved context should cover the golden answer",
        ),
        QualityRule(
            rule_id=QualityRuleId.MAX_TOOL_FAILURES,
            name="Maximum Tool Failures",
            operator=RuleOperator.MAXIMUM,
            threshold=float(thresholds.max_tool_failures),
            metric_aliases=[],
            severity=RuleSeverity.FAIL,
            description="Tool execution failures must stay within budget",
        ),
        QualityRule(
            rule_id=QualityRuleId.MAX_TOOL_LATENCY,
            name="Maximum Tool Latency",
            operator=RuleOperator.MAXIMUM,
            threshold=thresholds.max_tool_latency,
            metric_aliases=[],
            severity=RuleSeverity.WARNING,
            description="Per-tool / aggregate tool latency ceiling (ms)",
        ),
        QualityRule(
            rule_id=QualityRuleId.MAX_TOTAL_LATENCY,
            name="Maximum Overall Latency",
            operator=RuleOperator.MAXIMUM,
            threshold=thresholds.max_total_latency,
            metric_aliases=[],
            severity=RuleSeverity.WARNING,
            description="End-to-end latency ceiling (ms)",
        ),
        QualityRule(
            rule_id=QualityRuleId.MAX_COST,
            name="Maximum Cost",
            operator=RuleOperator.MAXIMUM,
            threshold=thresholds.max_cost,
            metric_aliases=[],
            severity=RuleSeverity.WARNING,
            description="Estimated USD cost ceiling per evaluation",
        ),
        QualityRule(
            rule_id=QualityRuleId.MIN_OVERALL_SCORE,
            name="Minimum Overall Score",
            operator=RuleOperator.MINIMUM,
            threshold=thresholds.pass_threshold,
            metric_aliases=["overall_score"],
            severity=RuleSeverity.FAIL,
            description="Mean metric score must meet the pass threshold",
        ),
    ]


__all__ = [
    "RuleOperator",
    "RuleSeverity",
    "QualityRuleId",
    "QualityRule",
    "QualityRuleResult",
    "QualityGateThresholds",
    "build_default_rules",
]
