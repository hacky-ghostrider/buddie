"""Evaluation framework — tool-independent metrics plus Sprint 10 automation.

Sprint 9: Metric ABC, registry, EvaluationService, structured reports.
Sprint 10: DeepEval adapters, LangSmith tracing, tool validation, golden
dataset loading, regression / benchmark runners (no LangGraph agents).
Sprint 10.2: EvaluationContext aggregate, ToolContract, ToolExecution,
ToolTraceMapper — architecture hardening for Sprint 11 agents.
Sprint 12: Quality gates, continuous evaluation, benchmark history.
"""

from app.evaluation.base import (
    EvaluationService,
    Metric,
    MetricRegistry,
    create_default_registry,
)
from app.evaluation.context import EvaluationContext
from app.evaluation.continuous import (
    ContinuousEvaluationResult,
    ContinuousEvaluationService,
)
from app.evaluation.exceptions import (
    EvaluationDisabledError,
    EvaluationError,
    InvalidEvaluationInputError,
    InvalidEvaluationReportError,
    MetricEvaluationError,
    MetricNotFoundError,
    MetricRegistrationError,
    MetricTimeoutError,
    NoRegisteredMetricsError,
)
from app.evaluation.metrics import AnswerLengthMetric, ContextCountMetric
from app.evaluation.models import GoldenExample, MetricResult
from app.evaluation.quality import (
    QualityDecision,
    QualityGateEngine,
    QualityStatus,
)
from app.evaluation.report import EvaluationReport
from app.evaluation.scenarios import (
    CANONICAL_DATASET_PATH,
    CANONICAL_SCENARIO_ID,
)
from app.evaluation.timeline import EvaluationTimeline, TimelineStage

__all__ = [
    "Metric",
    "MetricRegistry",
    "EvaluationService",
    "create_default_registry",
    "EvaluationReport",
    "EvaluationContext",
    "MetricResult",
    "GoldenExample",
    "AnswerLengthMetric",
    "ContextCountMetric",
    "EvaluationError",
    "EvaluationDisabledError",
    "NoRegisteredMetricsError",
    "MetricNotFoundError",
    "MetricRegistrationError",
    "MetricEvaluationError",
    "MetricTimeoutError",
    "InvalidEvaluationReportError",
    "InvalidEvaluationInputError",
    "CANONICAL_SCENARIO_ID",
    "CANONICAL_DATASET_PATH",
    "QualityDecision",
    "QualityGateEngine",
    "QualityStatus",
    "ContinuousEvaluationService",
    "ContinuousEvaluationResult",
    "EvaluationTimeline",
    "TimelineStage",
]
