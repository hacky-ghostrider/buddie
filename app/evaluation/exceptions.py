"""Domain exceptions for the evaluation stage.

Keep evaluation failures typed so callers (CLI, future API, CI harness) can
map them without inspecting string messages — same idea as checked domain
errors in a Java service layer.
"""


class EvaluationError(Exception):
    """Base error for all evaluation-framework failures."""


class EvaluationDisabledError(EvaluationError):
    """Raised when evaluation is invoked while ``ENABLE_EVALUATION`` is false."""


class NoRegisteredMetricsError(EvaluationError):
    """Raised when the registry has no enabled metrics to run."""


class MetricNotFoundError(EvaluationError):
    """Raised when a requested metric name is not registered."""


class MetricRegistrationError(EvaluationError):
    """Raised when metric registration is invalid (blank name, duplicate, …)."""


class MetricEvaluationError(EvaluationError):
    """Raised when a single metric fails during ``evaluate()``."""


class MetricTimeoutError(EvaluationError):
    """Raised when a metric exceeds ``METRIC_TIMEOUT``."""


class InvalidEvaluationReportError(EvaluationError):
    """Raised when an evaluation report cannot be built or fails validation."""


class InvalidEvaluationInputError(EvaluationError):
    """Raised when evaluation input (question / response) is invalid."""
