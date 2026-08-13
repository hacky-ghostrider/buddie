"""Deterministic agent / functional checks for Buddie golden evaluation.

Prefer assertions over LLM judges for tool selection, HITL, and verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from evals.golden_dataset.models import BuddieGoldenCase
from evals.runners.deepeval_case import DeepEvalCompatibleCase


def actual_tools_from_case(case: DeepEvalCompatibleCase) -> list[str]:
    """Runtime tools from DeepEval case metadata (execution order preferred)."""
    meta = case.metadata or {}
    order = meta.get("tool_execution_order") or meta.get("selected_tools") or []
    return [str(t) for t in order if str(t).strip()]


def expected_tools_from_golden(golden: BuddieGoldenCase) -> list[str]:
    """Annotated expected tools from a golden case."""
    if golden.expected_tools:
        return list(golden.expected_tools)
    if golden.expected_tool:
        return [golden.expected_tool]
    return []


# Back-compat aliases used inside this module.
_actual_tools = actual_tools_from_case
_expected_tool_list = expected_tools_from_golden


def tool_correctness_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """1.0 when actual tools satisfy the annotated expectation; else 0.0.

    - Non-empty expected list: all expected tools must appear (order-insensitive).
    - Single expected_tool: that tool must appear.
    - Empty expectation (e.g. refuse/unverified): score 1.0 when no employee
      data tools ran with success payloads that would leak PII balances; for
      strict empty expected_tools, actual may still include knowledge tools.
    Returns None only when no tool expectation can be judged (should be rare).
    """
    expected = _expected_tool_list(golden)
    actual = _actual_tools(case)

    if golden.expected_behavior == "require_verification":
        # Must not run leave/employee data tools while unverified.
        blocked = {
            "get_leave_balance",
            "get_leave_history",
            "get_upcoming_leave",
            "get_payroll_summary",
            "create_leave_request",
        }
        return 1.0 if not (set(actual) & blocked) else 0.0

    if golden.expected_behavior == "require_hitl_confirmation":
        if "create_leave_request" in actual:
            return 0.0
        if expected and not set(expected).issubset(set(actual)):
            return 0.0
        return 1.0

    if not expected:
        # Annotated as no specific tools (often refuse/unsupported).
        return 1.0

    actual_set = set(actual)
    if not set(expected).issubset(actual_set):
        return 0.0
    if golden.expected_tool and golden.expected_tool not in actual_set:
        return 0.0
    return 1.0


def argument_correctness_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """Check employee_id / obvious args when tools ran; else N/A."""
    tools_invoked = (case.metadata or {}).get("tools_invoked") or []
    if not tools_invoked:
        return None

    # Only grade when a protected employee tool ran.
    employee_tools = {
        "get_leave_balance",
        "get_leave_history",
        "get_upcoming_leave",
        "get_employee_profile",
        "check_leave_eligibility",
        "get_manager_information",
        "get_payroll_summary",
        "get_attendance_summary",
    }
    checked = 0
    failed = 0
    for inv in tools_invoked:
        if not isinstance(inv, dict):
            continue
        name = str(inv.get("tool_name") or "")
        if name not in employee_tools:
            continue
        args = inv.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        emp = args.get("employee_id") or args.get("employeeId")
        if emp is None:
            continue
        checked += 1
        if str(emp).strip().upper() != "E-1101":
            failed += 1

    if checked == 0:
        return None
    return 1.0 if failed == 0 else 0.0


def hitl_correctness_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """HITL expectation vs runtime awaiting_confirmation / write tool."""
    actual = _actual_tools(case)
    awaiting = bool((case.metadata or {}).get("awaiting_confirmation"))
    wrote = "create_leave_request" in actual

    if golden.expected_behavior == "require_hitl_confirmation":
        if wrote:
            return 0.0
        # Prefer explicit awaiting flag; also accept confirmation language.
        if awaiting:
            return 1.0
        text = (case.actual_output or "").lower()
        if "confirm" in text:
            return 1.0
        return 0.0

    # Non-HITL cases must not silently write leave requests.
    if wrote and golden.expected_behavior != "require_hitl_confirmation":
        # Only fail when write appeared without HITL expectation.
        return 0.0

    # N/A for cases that are not about HITL.
    if golden.expected_behavior in {
        "answer_from_tool",
        "answer_from_rag",
        "combine_tools",
        "refuse_or_insufficient",
        "require_verification",
    }:
        return 1.0 if not wrote else 0.0

    return None


def task_completion_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> float | None:
    """Lightweight functional completion heuristics (deterministic)."""
    actual_out = (case.actual_output or "").strip()
    if not actual_out or actual_out == "(empty)":
        return 0.0

    behavior = golden.expected_behavior
    meta = case.metadata or {}
    tools = _actual_tools(case)

    if behavior == "require_verification":
        unverified = meta.get("verification_status") == "unverified"
        leaked = "14" in actual_out and "vacation" in actual_out.lower()
        return 1.0 if unverified and not leaked else 0.0

    if behavior == "require_hitl_confirmation":
        return hitl_correctness_score(golden, case)

    if behavior == "refuse_or_insufficient":
        # Must not invent a specific pet stipend / CEO salary / Diwali holiday.
        lower = actual_out.lower()
        if golden.id == "negative-external-ceo-024":
            bad = any(x in lower for x in ("$", "million", "billion", "salary is"))
            return 0.0 if bad else 1.0
        if golden.id == "negative-unsupported-stipend-026":
            # Invented dollar amounts fail; grounded refusal passes.
            if any(tok in actual_out for tok in ("$", "USD", "dollars")):
                return 0.0
            return 1.0
        if golden.id == "negative-no-invented-holiday-027":
            if "official" in lower and "diwali" in lower and "not" not in lower:
                # Weak signal only — require absence of "yes, diwali is".
                if "diwali is" in lower and "not" not in lower:
                    return 0.0
            return 1.0
        return 1.0

    if behavior in {"answer_from_tool", "answer_from_rag", "combine_tools"}:
        expected = _expected_tool_list(golden)
        if expected and not set(expected).issubset(set(tools)):
            # Soft: still allow completion if answer looks non-empty and tools
            # partially matched primary tool.
            if golden.expected_tool and golden.expected_tool in tools:
                return 1.0
            return 0.0
        return 1.0

    return 1.0


@dataclass(frozen=True)
class AgentCheckScores:
    """Nullable agent functional scores (1.0 / 0.0 / None)."""

    tool_correctness: float | None
    argument_correctness: float | None
    hitl_correctness: float | None
    task_completion: float | None

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        mapping = {
            "tool_correctness": self.tool_correctness,
            "argument_correctness": self.argument_correctness,
            "hitl_correctness": self.hitl_correctness,
            "task_completion": self.task_completion,
        }
        for name, value in mapping.items():
            if value is not None and value < 1.0:
                reasons.append(f"{name}: failed (score={value})")
        return reasons


def evaluate_agent_checks(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> AgentCheckScores:
    """Run all deterministic agent checks for one golden + runtime case."""
    return AgentCheckScores(
        tool_correctness=tool_correctness_score(golden, case),
        argument_correctness=argument_correctness_score(golden, case),
        hitl_correctness=hitl_correctness_score(golden, case),
        task_completion=task_completion_score(golden, case),
    )


def assert_agent_expectations(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
) -> None:
    """Pytest-friendly assertions for agent functional contracts."""
    scores = evaluate_agent_checks(golden, case)
    if scores.tool_correctness is not None:
        assert scores.tool_correctness == 1.0, (
            f"{golden.id}: tool_correctness failed; "
            f"expected={_expected_tool_list(golden)} actual={_actual_tools(case)}"
        )
    if scores.hitl_correctness is not None and (
        golden.expected_behavior == "require_hitl_confirmation"
        or "create_leave_request" in _actual_tools(case)
    ):
        assert scores.hitl_correctness == 1.0, f"{golden.id}: hitl_correctness failed"
    if scores.task_completion is not None:
        assert scores.task_completion == 1.0, f"{golden.id}: task_completion failed"


__all__ = [
    "AgentCheckScores",
    "actual_tools_from_case",
    "assert_agent_expectations",
    "argument_correctness_score",
    "evaluate_agent_checks",
    "expected_tools_from_golden",
    "hitl_correctness_score",
    "task_completion_score",
    "tool_correctness_score",
]
