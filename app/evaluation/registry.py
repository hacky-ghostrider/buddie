"""Metric registry — plugin catalog for evaluation metrics.

Registries improve extensibility: new metrics (including future DeepEval /
RAGAS wrappers) register by name without editing ``EvaluationService``.
This is the same pattern as a Spring bean registry or the document-loader
extension map from Sprint 2.1.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from app.evaluation.exceptions import (
    MetricNotFoundError,
    MetricRegistrationError,
    NoRegisteredMetricsError,
)
from app.evaluation.metrics.base import Metric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Register, enable/disable, and resolve evaluation metrics by name.

    Responsibilities:
        - Register metric instances under a stable name.
        - Enable / disable metrics without unregistering them.
        - Retrieve metrics for the evaluation service.
        - Support a plugin-style architecture (register at startup / tests).

    Args:
        metrics: Optional initial metrics to register (all enabled).
    """

    def __init__(self, metrics: Iterable[Metric] | None = None) -> None:
        self._metrics: dict[str, Metric] = {}
        self._enabled: dict[str, bool] = {}
        if metrics is not None:
            for metric in metrics:
                self.register(metric)

    def register(self, metric: Metric, *, enabled: bool = True) -> None:
        """Register or replace a metric instance.

        Args:
            metric: Concrete ``Metric`` implementation.
            enabled: Whether the metric participates in evaluation runs.

        Raises:
            MetricRegistrationError: Blank name or ``metric`` is not a Metric.
        """
        if not isinstance(metric, Metric):
            raise MetricRegistrationError(
                f"Expected Metric instance, got {type(metric).__name__}"
            )
        name = metric.name().strip()
        if not name:
            raise MetricRegistrationError("Metric name must be a non-empty string")

        replacing = name in self._metrics
        self._metrics[name] = metric
        self._enabled[name] = enabled
        logger.info(
            "Metric %s registry: name=%s enabled=%s",
            "replaced" if replacing else "registered",
            name,
            enabled,
        )

    def unregister(self, name: str) -> None:
        """Remove a metric from the registry.

        Args:
            name: Registered metric name.

        Raises:
            MetricNotFoundError: Name is not registered.
        """
        key = name.strip()
        if key not in self._metrics:
            raise MetricNotFoundError(f"Metric '{name}' is not registered")
        del self._metrics[key]
        del self._enabled[key]
        logger.info("Metric unregistered: name=%s", key)

    def enable(self, name: str) -> None:
        """Enable a registered metric.

        Args:
            name: Registered metric name.

        Raises:
            MetricNotFoundError: Name is not registered.
        """
        key = self._require_registered(name)
        self._enabled[key] = True
        logger.info("Metric enabled: name=%s", key)

    def disable(self, name: str) -> None:
        """Disable a registered metric without removing it.

        Args:
            name: Registered metric name.

        Raises:
            MetricNotFoundError: Name is not registered.
        """
        key = self._require_registered(name)
        self._enabled[key] = False
        logger.info("Metric disabled: name=%s", key)

    def get(self, name: str) -> Metric:
        """Return a registered metric by name.

        Args:
            name: Registered metric name.

        Returns:
            The metric instance.

        Raises:
            MetricNotFoundError: Name is not registered.
        """
        key = self._require_registered(name)
        return self._metrics[key]

    def is_enabled(self, name: str) -> bool:
        """Return whether a registered metric is enabled.

        Args:
            name: Registered metric name.

        Returns:
            ``True`` if enabled.

        Raises:
            MetricNotFoundError: Name is not registered.
        """
        key = self._require_registered(name)
        return self._enabled[key]

    def list_registered(self) -> list[str]:
        """Return all registered metric names (enabled and disabled)."""
        return sorted(self._metrics)

    def list_enabled(self) -> list[str]:
        """Return names of currently enabled metrics."""
        return sorted(name for name, on in self._enabled.items() if on)

    def get_enabled_metrics(self) -> list[Metric]:
        """Return enabled metric instances in stable name order.

        Returns:
            Enabled metrics ready for evaluation.

        Raises:
            NoRegisteredMetricsError: No metrics are enabled.
        """
        enabled = [
            self._metrics[name]
            for name in self.list_enabled()
        ]
        if not enabled:
            raise NoRegisteredMetricsError(
                "No enabled metrics in the registry; register or enable at least one"
            )
        return enabled

    def as_mapping(self) -> Mapping[str, Metric]:
        """Return a read-only view of registered metrics."""
        return dict(self._metrics)

    def _require_registered(self, name: str) -> str:
        """Normalize and assert that ``name`` exists in the registry."""
        key = name.strip()
        if key not in self._metrics:
            raise MetricNotFoundError(
                f"Metric '{name}' is not registered. "
                f"Known: {sorted(self._metrics)}"
            )
        return key
