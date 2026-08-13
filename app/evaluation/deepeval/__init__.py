"""DeepEval adapter package — vendor SDK stays behind ``Metric`` implementations.

Business logic and ``EvaluationService`` must never import DeepEval types
directly. Register adapters via ``MetricRegistry`` instead.
"""

from app.evaluation.deepeval.adapter import (
    DeepEvalMetricAdapter,
    create_deepeval_metrics,
)
from app.evaluation.deepeval.metrics import DeepEvalMetricName

__all__ = [
    "DeepEvalMetricAdapter",
    "DeepEvalMetricName",
    "create_deepeval_metrics",
]
