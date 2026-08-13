"""Abstract metric contract for the evaluation framework.

Every concrete metric (placeholders today, DeepEval / RAGAS adapters later)
implements this interface so ``EvaluationService`` stays vendor-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.evaluation.models import EvaluationContext, MetricResult


class Metric(ABC):
    """Strategy interface for a single evaluation metric.

    Analogous to a Java interface with one scoring method: the orchestrator
    only knows ``evaluate`` / ``name`` / ``description``. Implementations may
    wrap DeepEval, RAGAS, or simple heuristics without changing callers.
    """

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> MetricResult:
        """Score one evaluation context.

        Args:
            context: Question, answer, retrieved docs, and optional goldens.

        Returns:
            Structured ``MetricResult`` with score in ``[0, 1]``.

        Raises:
            MetricEvaluationError: When scoring fails in a domain-specific way.
        """

    @abstractmethod
    def name(self) -> str:
        """Return the stable metric identifier used by the registry.

        Returns:
            Unique, non-empty metric name (e.g. ``answer_length``).
        """

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable explanation of what this metric measures.

        Returns:
            Short description suitable for logs and reports.
        """
