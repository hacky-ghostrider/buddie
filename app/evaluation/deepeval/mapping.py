"""Map domain evaluation models to / from DeepEval payloads.

WHY this module exists
----------------------
DeepEval uses ``LLMTestCase`` and vendor-specific field names
(``actual_output``, ``retrieval_context``, ``context``). Our domain uses
``EvaluationContext`` / ``MetricResult``. Mapping here keeps the adapter
thin and prevents DeepEval field names from leaking into the rest of the
platform — same idea as a DTO mapper between a Java domain and a SOAP SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.evaluation.deepeval.metrics import (
    METRICS_INVERT_SCORE,
    DeepEvalMetricName,
)
from app.evaluation.models import EvaluationContext, MetricResult

logger = logging.getLogger(__name__)


class SupportsLLMTestCase(Protocol):
    """Structural type for DeepEval ``LLMTestCase`` (avoids hard import)."""

    input: str
    actual_output: str
    expected_output: str | None
    retrieval_context: list[str] | None
    context: list[str] | None


def retrieval_texts(context: EvaluationContext) -> list[str]:
    """Extract retrieved chunk texts for DeepEval context fields.

    Args:
        context: Domain evaluation snapshot.

    Returns:
        Ordered list of retrieved document texts.
    """
    return [doc.text for doc in context.retrieved_documents]


def to_deepeval_test_case_kwargs(context: EvaluationContext) -> dict[str, Any]:
    """Build keyword arguments suitable for constructing ``LLMTestCase``.

    HallucinationMetric historically prefers ``context``; faithfulness /
    precision / recall prefer ``retrieval_context``. We populate both with
    the same retrieved texts so one mapping serves all metrics.

    Args:
        context: Domain evaluation snapshot.

    Returns:
        Dict of constructor kwargs (not a DeepEval instance).
    """
    texts = retrieval_texts(context)
    kwargs: dict[str, Any] = {
        "input": context.question,
        "actual_output": context.answer,
        "retrieval_context": texts,
        "context": texts,
    }
    if context.expected_answer is not None:
        kwargs["expected_output"] = context.expected_answer
    return kwargs


def normalize_deepeval_score(
    metric_name: DeepEvalMetricName | str,
    raw_score: float | None,
) -> float:
    """Normalize a DeepEval score into our ``[0, 1]`` higher-is-better scale.

    DeepEval's ``HallucinationMetric`` returns higher scores when more
    hallucination is detected. Our platform convention (and interview-safe
    rule) is **higher is always better**, so we invert that metric.

    Args:
        metric_name: Metric identity.
        raw_score: Score from DeepEval (may be ``None`` on failure).

    Returns:
        Clamped score in ``[0, 1]``.
    """
    if raw_score is None:
        return 0.0
    score = float(raw_score)
    name = (
        metric_name
        if isinstance(metric_name, DeepEvalMetricName)
        else DeepEvalMetricName(str(metric_name))
    )
    if name in METRICS_INVERT_SCORE:
        score = 1.0 - score
    return max(0.0, min(1.0, score))


def to_metric_result(
    *,
    metric_name: DeepEvalMetricName | str,
    raw_score: float | None,
    passed: bool | None,
    pass_threshold: float,
    reason: str | None = None,
    latency_ms: float = 0.0,
    error: str | None = None,
    extra_details: dict[str, Any] | None = None,
) -> MetricResult:
    """Convert a DeepEval measurement into a domain ``MetricResult``.

    Args:
        metric_name: Registry / report metric name.
        raw_score: Vendor score before normalization.
        passed: Vendor pass flag; recomputed from normalized score when None.
        pass_threshold: Per-metric pass cut-off.
        reason: Optional DeepEval reason / explanation string.
        latency_ms: Measurement latency.
        error: Optional error message.
        extra_details: Additional detail payload.

    Returns:
        Validated ``MetricResult``.
    """
    name = str(metric_name)
    score = normalize_deepeval_score(metric_name, raw_score)
    if error:
        return MetricResult(
            name=name,
            score=0.0,
            passed=False,
            details=dict(extra_details or {}),
            error=error,
            latency_ms=latency_ms,
        )

    details: dict[str, Any] = {
        "raw_score": raw_score,
        "normalized_score": score,
        "pass_threshold": pass_threshold,
        "provider": "deepeval",
    }
    if reason:
        details["reason"] = reason
    if extra_details:
        details.update(extra_details)

    metric_passed = passed if passed is not None else score >= pass_threshold
    return MetricResult(
        name=name,
        score=round(score, 6),
        passed=bool(metric_passed),
        details=details,
        error=None,
        latency_ms=latency_ms,
    )


__all__ = [
    "SupportsLLMTestCase",
    "retrieval_texts",
    "to_deepeval_test_case_kwargs",
    "normalize_deepeval_score",
    "to_metric_result",
]
