"""Tool workflow metrics — ordering, success rate, multi-tool completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evals.golden_dataset.models import BuddieGoldenCase
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

_WRITE_TOOLS = frozenset({"create_leave_request"})


def _tools_invoked(case: DeepEvalCompatibleCase) -> list[dict[str, Any]]:
    raw = (case.metadata or {}).get("tools_invoked") or []
    return [item for item in raw if isinstance(item, dict)]


def _is_success(inv: dict[str, Any]) -> bool:
    status = str(inv.get("status") or "").lower()
    if status in {"success", "ok", "pass"}:
        return True
    if status in {"failed", "failure", "error", "timeout", "cancelled"}:
        return False
    if inv.get("success") is False:
        return False
    if inv.get("error"):
        return False
    return True


def tool_ordering_correctness_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """1.0 when actual tool order preserves expected relative order."""
    from evals.metrics.agent_checks import (
        actual_tools_from_case,
        expected_tools_from_golden,
    )

    expected = expected_tools_from_golden(golden)
    if len(expected) < 2:
        return None

    actual_order = actual_tools_from_case(case)
    if not actual_order:
        return 0.0

    # Subsequence check: expected tools appear in the same relative order.
    exp_idx = 0
    for tool in actual_order:
        if exp_idx < len(expected) and tool == expected[exp_idx]:
            exp_idx += 1
    if exp_idx < len(expected):
        return 0.0
    return 1.0


def tool_call_success_rate(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """Fraction of invoked tools that succeeded; ``None`` when no tools ran."""
    invoked = _tools_invoked(case)
    if not invoked:
        return None

    successes = sum(1 for inv in invoked if _is_success(inv))
    return round(successes / len(invoked), 6)


def multi_tool_workflow_success_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """1.0 when multi-tool expectations are met with successful ordered calls."""
    from evals.metrics.agent_checks import (
        actual_tools_from_case,
        expected_tools_from_golden,
    )

    expected = expected_tools_from_golden(golden)
    if len(expected) < 2 or golden.expected_behavior not in {
        "combine_tools",
        "require_hitl_confirmation",
    }:
        return None

    actual = set(actual_tools_from_case(case))
    if not set(expected).issubset(actual):
        return 0.0

    ordering = tool_ordering_correctness_score(golden, case)
    if ordering is not None and ordering < 1.0:
        return 0.0

    success_rate = tool_call_success_rate(golden, case)
    if success_rate is not None and success_rate < 1.0:
        return 0.0

    if golden.expected_behavior == "require_hitl_confirmation":
        if actual & _WRITE_TOOLS:
            return 0.0

    return 1.0


@dataclass(frozen=True)
class ToolWorkflowScores:
    """Nullable tool workflow scores."""

    tool_ordering_correctness: float | None
    tool_call_success_rate: float | None
    multi_tool_workflow_success: float | None

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.tool_ordering_correctness is not None and self.tool_ordering_correctness < 1.0:
            reasons.append(
                f"tool_ordering_correctness: failed "
                f"(score={self.tool_ordering_correctness})"
            )
        if self.tool_call_success_rate is not None and self.tool_call_success_rate < 1.0:
            reasons.append(
                f"tool_call_success_rate: failed "
                f"(score={self.tool_call_success_rate})"
            )
        if (
            self.multi_tool_workflow_success is not None
            and self.multi_tool_workflow_success < 1.0
        ):
            reasons.append(
                f"multi_tool_workflow_success: failed "
                f"(score={self.multi_tool_workflow_success})"
            )
        return reasons


def evaluate_tool_workflow(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> ToolWorkflowScores:
    """Run tool workflow checks for one case."""
    return ToolWorkflowScores(
        tool_ordering_correctness=tool_ordering_correctness_score(golden, case),
        tool_call_success_rate=tool_call_success_rate(golden, case),
        multi_tool_workflow_success=multi_tool_workflow_success_score(golden, case),
    )


def tool_failure_messages(case: DeepEvalCompatibleCase) -> list[str]:
    """Collect tool error messages from runtime metadata."""
    messages: list[str] = []
    for inv in _tools_invoked(case):
        err = inv.get("error")
        if err and str(err).strip():
            messages.append(f"{inv.get('tool_name')}: {err}")
    meta = case.metadata or {}
    for key in ("routing_error", "last_tool_error", "infrastructure_error"):
        val = meta.get(key)
        if val and str(val).strip():
            messages.append(f"{key}: {val}")
    mcp = meta.get("mcp")
    if isinstance(mcp, dict):
        last = mcp.get("last_error")
        if last and str(last).strip():
            messages.append(f"mcp.last_error: {last}")
    return messages


__all__ = [
    "ToolWorkflowScores",
    "evaluate_tool_workflow",
    "multi_tool_workflow_success_score",
    "tool_call_success_rate",
    "tool_failure_messages",
    "tool_ordering_correctness_score",
]
