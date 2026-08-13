"""Standard DeepEval metrics for Buddie evaluation.

Faithfulness, Answer Relevancy, Hallucination, Contextual Precision/Recall/
Relevancy. Judge model comes from ``BuddieDeepEvalConfig.model`` (Gemini via
DeepEval ``GeminiModel`` when configured).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from evals.metrics.applicability import llm_metric_skip_reason
from evals.metrics.config import (
    METRIC_ANSWER_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_HALLUCINATION,
    METRICS_INVERT_RAW_SCORE,
    BuddieDeepEvalConfig,
)
from evals.metrics.rate_limit import (
    call_with_gemini_rate_limit_retry,
    is_rate_limit_error,
)
from evals.metrics.results import MetricScoreResult

if TYPE_CHECKING:
    from evals.golden_dataset.models import BuddieGoldenCase
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

logger = logging.getLogger(__name__)


class MetricMeasureFn(Protocol):
    """Injectable measure callable for deterministic tests."""

    def __call__(
        self,
        metric_name: str,
        test_case: Any,
        *,
        threshold: float,
    ) -> MetricScoreResult:
        """Return a measured score for ``metric_name``."""


def _build_llm_test_case(case: DeepEvalCompatibleCase) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(**case.to_llm_test_case_kwargs())


def _require_judge_model(model: Any | None) -> Any:
    """Live DeepEval metrics must receive an explicit GeminiModel (no GPT default)."""
    if model is None:
        raise RuntimeError(
            "DeepEval LLM judge model is missing. Pass an explicit GeminiModel "
            "via BuddieDeepEvalConfig.model (set GOOGLE_API_KEY or GEMINI_API_KEY). "
            "Omitting model falls back to GPTModel / OPENAI_API_KEY."
        )
    return model


def _live_metric(metric_name: str, threshold: float, model: Any | None) -> Any:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )

    judge = _require_judge_model(model)
    # Always pass the GeminiModel object — never a bare model-id string.
    kwargs: dict[str, Any] = {"threshold": threshold, "model": judge}

    builders = {
        METRIC_FAITHFULNESS: FaithfulnessMetric,
        METRIC_ANSWER_RELEVANCY: AnswerRelevancyMetric,
        METRIC_HALLUCINATION: HallucinationMetric,
        METRIC_CONTEXTUAL_PRECISION: ContextualPrecisionMetric,
        METRIC_CONTEXTUAL_RECALL: ContextualRecallMetric,
        METRIC_CONTEXTUAL_RELEVANCY: ContextualRelevancyMetric,
    }
    builder = builders.get(metric_name)
    if builder is None:
        raise KeyError(f"Unsupported standard metric: {metric_name}")
    return builder(**kwargs)


def build_standard_llm_metrics(config: BuddieDeepEvalConfig) -> dict[str, Any]:
    """Construct all standard LLM metrics with the same explicit judge model."""
    names = (
        METRIC_FAITHFULNESS,
        METRIC_ANSWER_RELEVANCY,
        METRIC_HALLUCINATION,
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    )
    return {
        name: _live_metric(name, config.threshold_for(name), config.model)
        for name in names
    }


def _skipped(name: str, threshold: float, reason: str) -> MetricScoreResult:
    return MetricScoreResult(
        name=name,
        score=None,
        passed=None,
        threshold=threshold,
        skipped=True,
        skip_reason=reason,
    )


def _rate_limited(name: str, threshold: float, detail: str) -> MetricScoreResult:
    return MetricScoreResult(
        name=name,
        score=None,
        passed=None,
        threshold=threshold,
        rate_limited=True,
        error=f"RATE_LIMITED: {detail}",
    )


def _normalize_score(metric_name: str, raw_score: float | None) -> float | None:
    if raw_score is None:
        return None
    score = float(raw_score)
    if metric_name in METRICS_INVERT_RAW_SCORE:
        score = 1.0 - score
    return max(0.0, min(1.0, score))


def measure_standard_metric(
    metric_name: str,
    case: DeepEvalCompatibleCase,
    config: BuddieDeepEvalConfig,
    *,
    measure_fn: MetricMeasureFn | None = None,
    golden: BuddieGoldenCase | None = None,
) -> MetricScoreResult:
    """Measure one standard DeepEval metric against a compatible case.

    When retrieval_context is empty, metrics that require actual evidence are
    skipped (not fabricated from golden expected_context).

    Hallucination scores are inverted to higher-is-better (less hallucination).
    """
    threshold = config.threshold_for(metric_name)

    skip_reason = llm_metric_skip_reason(metric_name, case, golden)
    if skip_reason is not None:
        return _skipped(metric_name, threshold, skip_reason)

    if measure_fn is not None:
        test_case = type("SimpleTestCase", (), case.to_llm_test_case_kwargs())()
        return measure_fn(metric_name, test_case, threshold=threshold)

    started = time.perf_counter()
    try:
        test_case = _build_llm_test_case(case)
        metric = _live_metric(metric_name, threshold, config.model)

        def _measure() -> None:
            metric.measure(test_case)

        call_with_gemini_rate_limit_retry(_measure)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_score = getattr(metric, "score", None)
        reason = getattr(metric, "reason", None)
        score = _normalize_score(
            metric_name,
            float(raw_score) if raw_score is not None else None,
        )
        # Recompute pass from normalized score so inverted metrics stay consistent.
        passed = score is not None and score >= threshold
        logger.debug(
            "Measured %s case=%s score=%s passed=%s latency_ms=%.1f",
            metric_name,
            case.case_id,
            score,
            passed,
            latency_ms,
        )
        return MetricScoreResult(
            name=metric_name,
            score=round(score, 6) if score is not None else None,
            passed=passed,
            threshold=threshold,
            reason=str(reason) if reason else None,
        )
    except Exception as exc:  # noqa: BLE001 — isolate vendor/LLM failures
        if is_rate_limit_error(exc):
            logger.warning(
                "Standard metric rate limited: metric=%s case=%s",
                metric_name,
                case.case_id,
            )
            return _rate_limited(metric_name, threshold, str(exc))
        logger.exception(
            "Standard metric failed: metric=%s case=%s",
            metric_name,
            case.case_id,
        )
        return MetricScoreResult(
            name=metric_name,
            score=None,
            passed=False,
            threshold=threshold,
            error=f"{metric_name} failed: {exc}",
        )


def measure_all_standard_metrics(
    case: DeepEvalCompatibleCase,
    config: BuddieDeepEvalConfig,
    *,
    measure_fn: MetricMeasureFn | None = None,
    golden: BuddieGoldenCase | None = None,
) -> dict[str, MetricScoreResult]:
    """Run the standard DeepEval metrics for one case."""
    names = (
        METRIC_FAITHFULNESS,
        METRIC_ANSWER_RELEVANCY,
        METRIC_HALLUCINATION,
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    )
    return {
        name: measure_standard_metric(
            name, case, config, measure_fn=measure_fn, golden=golden
        )
        for name in names
    }


__all__ = [
    "MetricMeasureFn",
    "build_standard_llm_metrics",
    "measure_all_standard_metrics",
    "measure_standard_metric",
]
