"""Domain exceptions for quality gates and continuous evaluation."""

from __future__ import annotations

from app.evaluation.exceptions import EvaluationError


class QualityGateError(EvaluationError):
    """Base error for quality-gate failures."""


class QualityGateDisabledError(QualityGateError):
    """Raised when quality gates are invoked while disabled."""


class InvalidQualityRuleError(QualityGateError):
    """Raised when a quality rule configuration is invalid."""


class InvalidQualityDecisionError(QualityGateError):
    """Raised when a quality decision cannot be constructed."""


class BenchmarkHistoryError(QualityGateError):
    """Raised when benchmark history persistence or load fails."""


__all__ = [
    "QualityGateError",
    "QualityGateDisabledError",
    "InvalidQualityRuleError",
    "InvalidQualityDecisionError",
    "BenchmarkHistoryError",
]
