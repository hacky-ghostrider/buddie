"""Placeholder metric: scores retrieval by retrieved-document count.

Not a relevance metric — only verifies that the evaluation pipeline can
inspect ``retrieved_documents``. Hit-rate / MRR / nDCG come later.
"""

from __future__ import annotations

from app.evaluation.exceptions import MetricEvaluationError
from app.evaluation.metrics.base import Metric
from app.evaluation.models import EvaluationContext, MetricResult


class ContextCountMetric(Metric):
    """Score whether enough documents were retrieved for grounding.

    Score is ``min(1.0, count / min_documents)``.

    Args:
        min_documents: Minimum expected retrieved documents (``> 0``).
        pass_threshold: Per-metric pass cut-off in ``[0, 1]``.
    """

    def __init__(
        self,
        *,
        min_documents: int = 1,
        pass_threshold: float = 0.7,
    ) -> None:
        if min_documents <= 0:
            raise ValueError("min_documents must be a positive integer")
        if pass_threshold < 0.0 or pass_threshold > 1.0:
            raise ValueError("pass_threshold must be between 0 and 1 inclusive")
        self._min_documents = min_documents
        self._pass_threshold = pass_threshold

    def name(self) -> str:
        """Return the registry key for this metric."""
        return "context_count"

    def description(self) -> str:
        """Return a short human-readable description."""
        return (
            f"Placeholder metric: retrieved document count vs minimum of "
            f"{self._min_documents}"
        )

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        """Compute document-count score for ``context.retrieved_documents``.

        Args:
            context: Evaluation snapshot.

        Returns:
            ``MetricResult`` with score in ``[0, 1]``.

        Raises:
            MetricEvaluationError: Unexpected scoring failure.
        """
        try:
            count = len(context.retrieved_documents)
            score = min(1.0, count / float(self._min_documents))
            return MetricResult(
                name=self.name(),
                score=round(score, 6),
                passed=score >= self._pass_threshold,
                details={
                    "retrieved_count": count,
                    "min_documents": self._min_documents,
                    "pass_threshold": self._pass_threshold,
                },
            )
        except Exception as exc:  # noqa: BLE001 — domain wrap for callers
            raise MetricEvaluationError(
                f"ContextCountMetric failed: {exc}"
            ) from exc
