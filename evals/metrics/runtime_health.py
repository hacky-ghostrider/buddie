"""Runtime health metrics — tool/API failures and graceful degradation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.golden_dataset.models import BuddieGoldenCase
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

_RAW_ERROR_MARKERS = (
    "traceback",
    "exception",
    "stack trace",
    "http 5",
    "http 4",
    "connection refused",
    "timeouterror",
    "jsondecodeerror",
    "keyerror",
    "attributeerror",
    "none type",
    "internal server error",
)


def runtime_graceful_degradation_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """1.0 when tool/API failures are surfaced without raw stack traces."""
    from evals.metrics.tool_workflow import tool_failure_messages

    failures = tool_failure_messages(case)
    if not failures:
        return None

    answer = (case.actual_output or "").lower()
    if not answer or answer == "(empty)":
        return 0.0

    for marker in _RAW_ERROR_MARKERS:
        if marker in answer:
            return 0.0

    if re.search(r"\bE\d{3}\b", answer):
        return 0.0

    return 1.0


def runtime_empty_response_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """1.0 when agent returned a non-empty answer (unless failure is expected)."""
    if golden.category == "adversarial_security" and golden.id.endswith("-empty-query-038"):
        return None
    actual = (case.actual_output or "").strip()
    if not actual or actual == "(empty)":
        return 0.0
    return 1.0


@dataclass(frozen=True)
class RuntimeHealthScores:
    """Nullable runtime health scores."""

    runtime_graceful_degradation: float | None
    runtime_empty_response: float | None
    tool_call_success_rate: float | None

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        mapping = {
            "runtime_graceful_degradation": self.runtime_graceful_degradation,
            "runtime_empty_response": self.runtime_empty_response,
            "tool_call_success_rate": self.tool_call_success_rate,
        }
        for name, value in mapping.items():
            if value is not None and value < 1.0:
                reasons.append(f"{name}: failed (score={value})")
        return reasons

    def has_tool_or_api_failure(self) -> bool:
        return bool(
            self.runtime_graceful_degradation is not None
            or (
                self.tool_call_success_rate is not None
                and self.tool_call_success_rate < 1.0
            )
        )


def evaluate_runtime_health(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> RuntimeHealthScores:
    """Score runtime failure handling for one case."""
    from evals.metrics.tool_workflow import tool_call_success_rate

    return RuntimeHealthScores(
        runtime_graceful_degradation=runtime_graceful_degradation_score(
            golden, case
        ),
        runtime_empty_response=runtime_empty_response_score(golden, case),
        tool_call_success_rate=tool_call_success_rate(golden, case),
    )


__all__ = [
    "RuntimeHealthScores",
    "evaluate_runtime_health",
    "runtime_empty_response_score",
    "runtime_graceful_degradation_score",
]
