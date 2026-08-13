"""DeepEval ``Metric`` adapters — Strategy implementations for the registry.

WHY an adapter
--------------
``EvaluationService`` only knows ``Metric.evaluate(context) → MetricResult``.
DeepEval has its own test-case model, metric classes, and LLM-as-judge calls.
This adapter translates both ways so swapping DeepEval for RAGAS later does
not rewrite the evaluation service (Dependency Inversion).

HOW it works
------------
1. Map ``EvaluationContext`` → DeepEval ``LLMTestCase`` kwargs.
2. Instantiate / reuse the vendor metric object.
3. Call ``measure(test_case)`` (injectable for unit tests).
4. Map vendor score → domain ``MetricResult`` (invert hallucination).

TRADEOFFS
---------
- LLM-as-judge metrics are non-deterministic and cost money; gate with
  ``ENABLE_DEEPEVAL`` and prefer offline golden suites in CI with mocks.
- Soft-import keeps the package importable when DeepEval is not installed
  (tests inject a measure callable).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from app.evaluation.deepeval.mapping import (
    to_deepeval_test_case_kwargs,
    to_metric_result,
)
from app.evaluation.deepeval.metrics import (
    DEEPEVAL_METRIC_DESCRIPTIONS,
    DEFAULT_DEEPEVAL_METRICS,
    METRICS_REQUIRING_EXPECTED_ANSWER,
    DeepEvalMetricName,
)
from app.evaluation.exceptions import MetricEvaluationError
from app.evaluation.metrics.base import Metric
from app.evaluation.models import EvaluationContext, MetricResult

logger = logging.getLogger(__name__)


class _DeepEvalMetricLike(Protocol):
    """Minimal protocol for DeepEval metric instances used by the adapter."""

    score: float | None
    reason: str | None
    success: bool | None
    threshold: float

    def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the vendor metric against a test case."""


MeasureFn = Callable[[Any], _DeepEvalMetricLike]
"""Callable that accepts an ``LLMTestCase``-like object and returns a measured metric."""


class DeepEvalMetricAdapter(Metric):
    """Wrap one DeepEval metric behind the domain ``Metric`` interface.

    Args:
        metric_name: Stable registry key / report name.
        pass_threshold: Pass cut-off on the *normalized* score.
        measure_fn: Optional injectable measure function (tests / fakes).
            When omitted, the real DeepEval metric class is constructed.
        metric_factory: Optional factory returning a fresh DeepEval metric
            instance (used when ``measure_fn`` is not provided).
    """

    def __init__(
        self,
        metric_name: DeepEvalMetricName | str,
        *,
        pass_threshold: float = 0.7,
        measure_fn: MeasureFn | None = None,
        metric_factory: Callable[[], _DeepEvalMetricLike] | None = None,
    ) -> None:
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0 and 1 inclusive")
        self._metric_name = DeepEvalMetricName(str(metric_name))
        self._pass_threshold = pass_threshold
        self._measure_fn = measure_fn
        self._metric_factory = metric_factory

    def name(self) -> str:
        """Return the registry key for this DeepEval-backed metric."""
        return str(self._metric_name)

    def description(self) -> str:
        """Return a human-readable description of the metric."""
        return DEEPEVAL_METRIC_DESCRIPTIONS.get(
            self._metric_name,
            f"DeepEval metric: {self._metric_name}",
        )

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        """Score ``context`` using DeepEval (or an injected measure function).

        Args:
            context: Domain evaluation snapshot.

        Returns:
            Normalized ``MetricResult`` (higher is better).

        Raises:
            MetricEvaluationError: When mapping / vendor invocation fails hard.
        """
        if (
            self._metric_name in METRICS_REQUIRING_EXPECTED_ANSWER
            and not (context.expected_answer and context.expected_answer.strip())
        ):
            return MetricResult(
                name=self.name(),
                score=0.0,
                passed=False,
                details={"provider": "deepeval"},
                error=(
                    f"Metric '{self.name()}' requires expected_answer "
                    "(golden dataset field)"
                ),
                latency_ms=0.0,
            )

        started = time.perf_counter()
        try:
            measured = self._invoke_measure(context)
            latency_ms = (time.perf_counter() - started) * 1000.0
            raw_score = getattr(measured, "score", None)
            reason = getattr(measured, "reason", None)
            success = getattr(measured, "success", None)
            return to_metric_result(
                metric_name=self._metric_name,
                raw_score=float(raw_score) if raw_score is not None else None,
                passed=bool(success) if success is not None else None,
                pass_threshold=self._pass_threshold,
                reason=str(reason) if reason else None,
                latency_ms=latency_ms,
            )
        except MetricEvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001 — isolate vendor failures
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "DeepEval metric failed: name=%s error=%s",
                self.name(),
                exc,
            )
            return to_metric_result(
                metric_name=self._metric_name,
                raw_score=None,
                passed=False,
                pass_threshold=self._pass_threshold,
                latency_ms=latency_ms,
                error=f"DeepEval metric '{self.name()}' failed: {exc}",
            )

    def _invoke_measure(self, context: EvaluationContext) -> _DeepEvalMetricLike:
        """Build a test case and measure via injection or live DeepEval."""
        kwargs = to_deepeval_test_case_kwargs(context)

        if self._measure_fn is not None:
            test_case = _SimpleTestCase(**kwargs)
            return self._measure_fn(test_case)

        test_case = self._build_live_test_case(kwargs)
        metric = self._resolve_live_metric()
        metric.measure(test_case)
        return metric

    def _resolve_live_metric(self) -> _DeepEvalMetricLike:
        """Construct the real DeepEval metric instance."""
        if self._metric_factory is not None:
            return self._metric_factory()
        return _default_metric_factory(self._metric_name, self._pass_threshold)()

    @staticmethod
    def _build_live_test_case(kwargs: dict[str, Any]) -> Any:
        """Import and construct DeepEval ``LLMTestCase`` lazily."""
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:  # pragma: no cover - env dependent
            raise MetricEvaluationError(
                "deepeval is not installed; add the 'deepeval' dependency "
                "or inject measure_fn for tests"
            ) from exc
        return LLMTestCase(**kwargs)


class _SimpleTestCase:
    """Lightweight stand-in for ``LLMTestCase`` used by injected measure fns."""

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def _default_metric_factory(
    metric_name: DeepEvalMetricName,
    threshold: float,
) -> Callable[[], _DeepEvalMetricLike]:
    """Return a factory that constructs the matching DeepEval metric class."""

    def factory() -> _DeepEvalMetricLike:
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
                HallucinationMetric,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise MetricEvaluationError(
                "deepeval is not installed; cannot construct live metrics"
            ) from exc

        mapping: dict[DeepEvalMetricName, Callable[[], _DeepEvalMetricLike]] = {
            DeepEvalMetricName.FAITHFULNESS: lambda: FaithfulnessMetric(
                threshold=threshold
            ),
            DeepEvalMetricName.HALLUCINATION: lambda: HallucinationMetric(
                threshold=threshold
            ),
            DeepEvalMetricName.ANSWER_RELEVANCY: lambda: AnswerRelevancyMetric(
                threshold=threshold
            ),
            DeepEvalMetricName.CONTEXTUAL_PRECISION: (
                lambda: ContextualPrecisionMetric(threshold=threshold)
            ),
            DeepEvalMetricName.CONTEXTUAL_RECALL: lambda: ContextualRecallMetric(
                threshold=threshold
            ),
        }
        builder = mapping.get(metric_name)
        if builder is None:
            raise MetricEvaluationError(f"Unsupported DeepEval metric: {metric_name}")
        return builder()

    return factory


def create_deepeval_metrics(
    *,
    pass_threshold: float = 0.7,
    metric_names: Sequence[DeepEvalMetricName | str] | None = None,
    measure_fns: dict[str, MeasureFn] | None = None,
) -> list[DeepEvalMetricAdapter]:
    """Build the standard suite of DeepEval adapters.

    Args:
        pass_threshold: Shared pass cut-off for normalized scores.
        metric_names: Subset to create; defaults to the full Sprint 10 suite.
        measure_fns: Optional map of metric name → injectable measure function.

    Returns:
        List of ``DeepEvalMetricAdapter`` instances ready for registration.
    """
    names = list(metric_names) if metric_names is not None else list(DEFAULT_DEEPEVAL_METRICS)
    adapters: list[DeepEvalMetricAdapter] = []
    for raw_name in names:
        name = DeepEvalMetricName(str(raw_name))
        measure = (measure_fns or {}).get(str(name))
        adapters.append(
            DeepEvalMetricAdapter(
                name,
                pass_threshold=pass_threshold,
                measure_fn=measure,
            )
        )
    logger.info(
        "Created DeepEval adapters: count=%d names=%s",
        len(adapters),
        [a.name() for a in adapters],
    )
    return adapters


__all__ = [
    "DeepEvalMetricAdapter",
    "MeasureFn",
    "create_deepeval_metrics",
]
