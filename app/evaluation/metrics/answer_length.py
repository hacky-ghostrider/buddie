"""Placeholder metric: scores answers by minimum character length.

Not a quality metric — only proves the framework plumbing. Real semantic /
faithfulness metrics arrive with DeepEval / RAGAS later.
"""

from __future__ import annotations

from app.evaluation.exceptions import MetricEvaluationError
from app.evaluation.metrics.base import Metric
from app.evaluation.models import EvaluationContext, MetricResult


class AnswerLengthMetric(Metric):
    """Score whether the generated answer meets a minimum length.

    Score is ``min(1.0, len(answer) / min_length)`` so short answers get a
    fractional score and sufficiently long answers score ``1.0``.

    Args:
        min_length: Minimum acceptable answer length in characters (``> 0``).
        pass_threshold: Per-metric pass cut-off in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        min_length: int = 20,
        pass_threshold: float = 0.7,
    ) -> None:
        if min_length <= 0:
            raise ValueError("min_length must be a positive integer")
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0 and 1 inclusive")
        self._min_length = min_length
        self._pass_threshold = pass_threshold

    def name(self) -> str:
        """Return the registry key for this metric."""
        return "answer_length"

    def description(self) -> str:
        """Return a short human-readable description."""
        return (
            f"Placeholder metric: answer length vs minimum of "
            f"{self._min_length} characters"
        )

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        """Compute length-based score for ``context.answer``.

        Args:
            context: Evaluation snapshot.

        Returns:
            ``MetricResult`` with score in ``[0, 1]``.

        Raises:
            MetricEvaluationError: Unexpected scoring failure.
        """
        try:
            length = len(context.answer.strip())
            score = min(1.0, length / float(self._min_length))
            return MetricResult(
                name=self.name(),
                score=round(score, 6),
                passed=score >= self._pass_threshold,
                details={
                    "answer_length": length,
                    "min_length": self._min_length,
                    "pass_threshold": self._pass_threshold,
                },
            )
        except Exception as exc:  # noqa: BLE001 — domain wrap for callers
            raise MetricEvaluationError(
                f"AnswerLengthMetric failed: {exc}"
            ) from exc
