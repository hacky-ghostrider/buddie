"""Buddie evaluation metrics package (Sprint 18–19)."""

from evals.metrics.annotations import (
    AnnotationCoverageReport,
    build_annotation_report,
    format_annotation_console,
)
from evals.metrics.config import (
    DEFAULT_PASS_THRESHOLD,
    PRIMARY_METRIC_NAMES,
    BuddieDeepEvalConfig,
    default_buddie_deepeval_config,
    gemini_judge_configured,
    gemini_judge_status,
)
from evals.metrics.results import (
    CaseEvaluationResult,
    MetricScoreResult,
    SuiteEvaluationReport,
)
from evals.metrics.retrieval import (
    compute_retrieval_metrics,
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "AnnotationCoverageReport",
    "DEFAULT_PASS_THRESHOLD",
    "PRIMARY_METRIC_NAMES",
    "BuddieDeepEvalConfig",
    "CaseEvaluationResult",
    "MetricScoreResult",
    "SuiteEvaluationReport",
    "build_annotation_report",
    "compute_retrieval_metrics",
    "default_buddie_deepeval_config",
    "format_annotation_console",
    "gemini_judge_configured",
    "gemini_judge_status",
    "hit_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
