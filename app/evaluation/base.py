"""Evaluation package base re-exports.

Mirrors ``app.generation.base`` / ``app.embeddings.base`` so callers can import
the primary abstractions from a stable path.
"""

from app.evaluation.evaluator import EvaluationService, create_default_registry
from app.evaluation.metrics.base import Metric
from app.evaluation.registry import MetricRegistry

__all__ = [
    "Metric",
    "MetricRegistry",
    "EvaluationService",
    "create_default_registry",
]
