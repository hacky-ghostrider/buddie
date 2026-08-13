"""Tool router — execute planner output and populate AgentState.

The router is the *foreman*: it receives ``PlannerOutput``, runs tools via
the registry, collects ``ToolExecution`` objects, and seeds fields that
later become ``EvaluationContext``. It does not invent tool policy —
contracts come from the planner / goldens.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.conversation import (
    GENERATION_FAILURE,
    RAG_FAILURE,
    TOOL_FAILURE,
    VERIFY_FAILED,
    VERIFY_PROMPT,
    sanitize_user_facing_answer,
)
from app.agent.exceptions import AgentRoutingError, AgentToolNotFoundError
from app.agent.models import PlannerOutput, ToolInvocation
from app.agent.state import AgentState
from app.agent.tools.base import ToolRegistry
from app.agent.tools.employee_tools import PROTECTED_EMPLOYEE_TOOLS
from app.evaluation.tool_validation.tool_contract import ToolContract
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionStatus,
)

logger = logging.getLogger(__name__)


class ToolRouter:
    """Execute planned tools in order and update agent state.

    Args:
        registry: Injected tool registry.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def route(
        self,
        planner_output: PlannerOutput,
        *,
        question: str,
        correlation_id: str,
        shared_context: dict[str, Any] | None = None,
    ) -> list[ToolExecution]:
        """Execute invocations from ``planner_output``.

        Args:
            planner_output: Structured planner decision.
            question: Original user question (shared with tools).
            correlation_id: Correlation id for logging.
            shared_context: Mutable context shared across tools (RAG cache).

        Returns:
            Ordered ``ToolExecution`` list.
        """
        context = shared_context if shared_context is not None else {}
        context.setdefault("question", question)

        invocations = list(planner_output.invocations)
        if not invocations:
            invocations = [
                ToolInvocation(tool_name=name, arguments={}, order=index)
                for index, name in enumerate(planner_output.execution_order)
            ]

        executions: list[ToolExecution] = []
        for index, invocation in enumerate(invocations):
            order = invocation.order if invocation.order is not None else index
            try:
                execution = self._registry.execute(
                    invocation.tool_name,
                    dict(invocation.arguments),
                    order=order,
                    correlation_id=correlation_id,
                    context=context,
                )
            except AgentToolNotFoundError as exc:
                logger.error(
                    "Tool missing during routing: correlation_id=%s tool=%s",
                    correlation_id,
                    invocation.tool_name,
                )
                execution = ToolExecution(
                    tool_name=invocation.tool_name,
                    arguments=dict(invocation.arguments),
                    status=ToolExecutionStatus.FAILED,
                    error=str(exc),
                    order=order,
                )
            executions.append(execution)
            logger.info(
                "Routed tool: correlation_id=%s tool=%s status=%s order=%s",
                correlation_id,
                execution.tool_name,
                execution.status.value,
                execution.order,
            )
        return executions

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph node entrypoint."""
        correlation_id = state.get("correlation_id", "")
        question = state.get("question", "")
        raw_plan = state.get("planner_output")
        if not raw_plan:
            raise AgentRoutingError("Router requires planner_output in state")

        try:
            planner_output = PlannerOutput.model_validate(raw_plan)
        except Exception as exc:  # noqa: BLE001
            raise AgentRoutingError(f"Invalid planner_output: {exc}") from exc

        shared_context: dict[str, Any] = {"question": question}
        metadata = dict(state.get("metadata") or {})
        # Bind verified session employee into tool context (never from LLM args).
        eid = metadata.get("verified_employee_id") or metadata.get("employee_id")
        if eid:
            shared_context["verified_employee_id"] = str(eid).strip().upper()
            shared_context["employee_id"] = shared_context["verified_employee_id"]
        shared_context["metadata"] = metadata
        if planner_output.pending_action:
            shared_context["pending_action"] = dict(planner_output.pending_action)
        # Confirmed write: only when planner explicitly scheduled create_leave_request.
        if any(
            inv.tool_name == "create_leave_request"
            and bool(inv.arguments.get("confirmed"))
            for inv in planner_output.invocations
        ):
            shared_context["leave_request_confirmed"] = True

        # Conversational / gate responses: no tools, direct answer only.
        if planner_output.direct_answer and not planner_output.execution_order:
            answer = sanitize_user_facing_answer(planner_output.direct_answer)
            logger.info(
                "Router direct answer: correlation_id=%s route=%s",
                correlation_id,
                planner_output.intent_route,
            )
            return {
                "tool_execution_history": [],
                "tool_results": [],
                "selected_tools": [],
                "tool_contracts": [],
                "final_answer": answer,
                "messages": [
                    {
                        "role": "assistant",
                        "content": answer,
                        "intent_route": planner_output.intent_route,
                    }
                ],
                "metadata": {
                    **metadata,
                    "intent_route": planner_output.intent_route,
                    "direct_answer": True,
                    "last_rag_response": None,
                    "verified_employee_id": shared_context.get("verified_employee_id"),
                    "pending_leave_request": None,
                    "awaiting_confirmation": False,
                },
            }

        try:
            executions = self.route(
                planner_output,
                question=question,
                correlation_id=correlation_id,
                shared_context=shared_context,
            )
        except AgentRoutingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AgentRoutingError(f"Tool routing failed: {exc}") from exc

        final_answer = _derive_final_answer(
            executions,
            shared_context,
            pending_action=planner_output.pending_action,
        )
        pending_leave = None
        awaiting = False
        if planner_output.pending_action and planner_output.pending_action.get(
            "awaiting_confirmation"
        ):
            pending_leave = dict(planner_output.pending_action)
            awaiting = True
        # Clear pending after successful write.
        if any(
            e.tool_name == "create_leave_request" and e.success for e in executions
        ):
            pending_leave = None
            awaiting = False

        return {
            "tool_execution_history": [
                e.model_dump(mode="json") for e in executions
            ],
            "tool_results": [e.output for e in executions],
            "selected_tools": [e.tool_name for e in executions],
            "tool_contracts": [
                c.model_dump(mode="json") for c in planner_output.tool_contracts
            ],
            "final_answer": final_answer,
            "messages": [
                {
                    "role": "tool",
                    "tool_name": e.tool_name,
                    "status": e.status.value,
                    "order": e.order,
                }
                for e in executions
            ],
            "metadata": {
                **metadata,
                "intent_route": planner_output.intent_route,
                "last_rag_response": _serialize_rag(shared_context),
                "verified_employee_id": shared_context.get("verified_employee_id"),
                "routing_error": _last_tool_error(executions),
                "pending_leave_request": pending_leave,
                "awaiting_confirmation": awaiting,
                "tool_execution_order": [e.tool_name for e in executions],
            },
        }


def _derive_final_answer(
    executions: list[ToolExecution],
    shared_context: dict[str, Any],
    *,
    pending_action: dict[str, Any] | None = None,
) -> str:
    """Pick the best final answer from tool outputs / shared context."""
    # Multi-tool combined responses (manager+holidays, eligibility, etc.).
    combined = _combine_multi_tool_answer(executions, shared_context)
    if combined:
        answer = sanitize_user_facing_answer(combined)
        if pending_action and pending_action.get("awaiting_confirmation"):
            answer = _append_confirmation_prompt(answer, pending_action)
        return answer

    if "summary" in shared_context and shared_context["summary"]:
        # Hybrid: prefer combining leave balance + policy summary when both exist.
        balance = None
        for execution in executions:
            if execution.tool_name == "get_leave_balance" and execution.success:
                balance = execution.output
                break
        summary = sanitize_user_facing_answer(str(shared_context["summary"]))
        if isinstance(balance, dict) and balance.get("leave_balance"):
            lb = balance["leave_balance"]
            return (
                f"Your current leave balance is vacation={lb.get('vacation')}, "
                f"sick={lb.get('sick')}, personal={lb.get('personal')}. "
                f"Policy: {summary}"
            )
        return summary

    for execution in reversed(executions):
        if not execution.success:
            continue
        output = execution.output
        if isinstance(output, dict):
            formatted = _format_employee_output(execution.tool_name, output)
            if formatted:
                return sanitize_user_facing_answer(formatted)
            for key in ("summary", "result", "answer"):
                if key in output and output[key] is not None:
                    return sanitize_user_facing_answer(str(output[key]))
            if "results" in output:
                return sanitize_user_facing_answer(str(output["results"]))
            # Fall back to compact JSON for structured employee payloads.
            if execution.tool_name.startswith("get_") or execution.tool_name == (
                "verify_employee"
            ):
                return sanitize_user_facing_answer(str(output))
        if isinstance(output, str) and output.strip():
            return sanitize_user_facing_answer(output)

    failed = [e for e in executions if not e.success]
    if failed:
        # Partial multi-tool failure: surface successes when available.
        successes = [e for e in executions if e.success]
        if successes:
            partial = _combine_multi_tool_answer(successes, shared_context)
            if partial:
                return sanitize_user_facing_answer(
                    partial
                    + "\n\nSome related information could not be retrieved right now."
                )
        return _friendly_failure_message(failed[-1])
    return ""


def _append_confirmation_prompt(
    answer: str,
    pending: dict[str, Any],
) -> str:
    leave_type = str(pending.get("leave_type") or "VACATION").title()
    start = pending.get("start_date") or "?"
    end = pending.get("end_date") or "?"
    days = pending.get("requested_days")
    day_label = f"{days:g}" if isinstance(days, (int, float)) else str(days or "?")
    return (
        f"{answer}\n\n"
        f"I can create a {leave_type} leave request for {day_label} day(s) "
        f"from {start} to {end}.\n"
        "Reply **confirm** to submit this request, or **cancel** to discard it."
    )


def _combine_multi_tool_answer(
    executions: list[ToolExecution],
    shared_context: dict[str, Any],
) -> str | None:
    """Build a combined answer when multiple employee tools succeeded."""
    by_name = {
        e.tool_name: e
        for e in executions
        if e.success and isinstance(e.output, dict)
    }
    if not by_name:
        return None

    # Manager + holidays
    if "get_manager_information" in by_name and (
        "get_holiday_calendar" in by_name or "get_upcoming_holidays" in by_name
    ):
        parts: list[str] = []
        mgr = _format_employee_output(
            "get_manager_information", by_name["get_manager_information"].output
        )
        if mgr:
            parts.append(mgr)
        holiday_tool = (
            "get_holiday_calendar"
            if "get_holiday_calendar" in by_name
            else "get_upcoming_holidays"
        )
        hol = _format_employee_output(holiday_tool, by_name[holiday_tool].output)
        if hol:
            parts.append(hol)
        return "\n\n".join(parts) if parts else None

    # Eligibility workflow
    if "check_leave_eligibility" in by_name:
        parts = []
        if "get_employee_profile" in by_name:
            profile = _format_employee_output(
                "get_employee_profile", by_name["get_employee_profile"].output
            )
            if profile:
                parts.append(profile)
        if "get_leave_balance" in by_name:
            bal = _format_employee_output(
                "get_leave_balance", by_name["get_leave_balance"].output
            )
            if bal:
                parts.append(bal)
        elig = _format_employee_output(
            "check_leave_eligibility", by_name["check_leave_eligibility"].output
        )
        if elig:
            parts.append(elig)
        if "search_company_policy" in by_name:
            policy = by_name["search_company_policy"].output
            summary = policy.get("summary") or shared_context.get("summary")
            if summary:
                parts.append(f"Policy: {sanitize_user_facing_answer(str(summary))}")
        return "\n\n".join(parts) if parts else None

    return None


def _friendly_failure_message(execution: ToolExecution) -> str:
    """Map tool failures to employee-safe copy (details stay in logs/metadata)."""
    error = (execution.error or "").lower()
    logger.warning(
        "Tool failure surfaced as friendly message: tool=%s error=%s",
        execution.tool_name,
        execution.error,
    )
    if execution.tool_name == "verify_employee":
        return VERIFY_FAILED
    if execution.tool_name in PROTECTED_EMPLOYEE_TOOLS:
        if "verif" in error or "required" in error:
            return VERIFY_PROMPT
        return TOOL_FAILURE
    if execution.tool_name in {"search_docs", "summarize", "search_company_policy"}:
        if any(
            token in error
            for token in ("openai", "api key", "timeout", "generation", "llm")
        ):
            return GENERATION_FAILURE
        return RAG_FAILURE
    return TOOL_FAILURE


def _last_tool_error(executions: list[ToolExecution]) -> str | None:
    """Preserve last tool error for developer / evaluation mode."""
    for execution in reversed(executions):
        if not execution.success and execution.error:
            return execution.error
    return None


def _format_employee_output(tool_name: str, output: dict[str, Any]) -> str | None:
    """Render common employee tool payloads as concise assistant text."""
    if tool_name == "get_leave_balance" and "leave_balance" in output:
        lb = output["leave_balance"]
        return (
            f"Leave balance for {output.get('employee_id')}: "
            f"vacation={lb.get('vacation')} days, "
            f"sick={lb.get('sick')} days, "
            f"personal={lb.get('personal')} days."
        )
    if tool_name == "get_leave_history" and "leave_history" in output:
        return _format_leave_history(output)
    if tool_name == "get_manager_information":
        return (
            f"Your manager is {output.get('manager')} "
            f"({output.get('department')} / {output.get('designation')})."
        )
    if tool_name in {"get_upcoming_holidays", "get_holiday_calendar"}:
        return _format_holidays(output)
    if tool_name == "check_leave_eligibility":
        eligible = output.get("eligible")
        leave_type = str(output.get("leave_type") or "leave").title()
        requested = output.get("requested_days")
        available = output.get("available_days")
        status = "eligible" if eligible else "not eligible"
        lines = [
            f"Leave eligibility ({leave_type}): {status}.",
            f"Requested {requested:g} day(s); available {available:g} day(s)."
            if isinstance(requested, (int, float))
            and isinstance(available, (int, float))
            else f"Requested {requested}; available {available}.",
        ]
        for reason in output.get("reasons") or []:
            lines.append(f"- {reason}")
        for note in output.get("policy_notes") or []:
            lines.append(f"Note: {note}")
        return "\n".join(lines)
    if tool_name == "create_leave_request":
        if output.get("created"):
            return str(output.get("message") or "Leave request created.")
        return "Leave request was not created."
    if tool_name == "search_company_policy" and output.get("summary"):
        return sanitize_user_facing_answer(str(output["summary"]))
    if tool_name == "get_pending_actions":
        actions = output.get("pending_actions") or []
        count = output.get("count", len(actions))
        if not actions:
            return "You have no pending HR actions."
        lines = [f"You have {count} pending HR action(s):"]
        for item in actions:
            if not isinstance(item, dict):
                continue
            due = item.get("due_date") or "—"
            status = item.get("status") or "PENDING"
            desc = item.get("description") or item.get("action_id") or "Action"
            lines.append(f"- {desc} (due {due}, {status})")
        return "\n".join(lines)
    if tool_name == "get_upcoming_leave":
        nxt = output.get("next_vacation")
        if isinstance(nxt, dict):
            return (
                f"Your next vacation is {nxt.get('start_date')} to "
                f"{nxt.get('end_date')} ({nxt.get('status')})."
            )
        return "You have no upcoming vacation on file."
    if tool_name == "get_employee_profile":
        return (
            f"{output.get('full_name')} ({output.get('employee_id')}) — "
            f"{output.get('designation')} in {output.get('department')}, "
            f"manager {output.get('manager')}."
        )
    if tool_name == "get_payroll_summary" and "payroll" in output:
        pay = output["payroll"]
        return (
            f"Payroll for {output.get('employee_id')}: "
            f"net {pay.get('monthly_net')} {pay.get('currency')} / month, "
            f"next pay date {pay.get('next_pay_date')} "
            f"({pay.get('payroll_status')})."
        )
    if tool_name == "verify_employee":
        if output.get("verified"):
            name = output.get("full_name")
            eid = output.get("employee_id")
            if name:
                return f"You're verified ✓ Welcome, {name} ({eid})."
            return f"You're verified ✓ ({eid})."
        return VERIFY_FAILED
    return None


def _format_holidays(output: dict[str, Any]) -> str:
    """Format upcoming holidays or a year calendar."""
    nxt = output.get("next_holiday")
    holidays = output.get("upcoming_holidays") or output.get("holidays") or []
    if isinstance(nxt, dict):
        lines = [
            f"Next company holiday: {nxt.get('holiday_name')} "
            f"on {nxt.get('date')} ({nxt.get('location')})."
        ]
        extras = [
            h
            for h in holidays
            if isinstance(h, dict)
            and (
                h.get("date") != nxt.get("date")
                or h.get("holiday_name") != nxt.get("holiday_name")
            )
        ]
        for item in extras[:4]:
            lines.append(
                f"- {item.get('holiday_name')} on {item.get('date')} "
                f"({item.get('location')})"
            )
        return "\n".join(lines)
    if holidays:
        country = output.get("country")
        year = output.get("year")
        title = "Company holidays"
        if country or year:
            title = f"Holiday calendar ({country or 'All'}"
            if year:
                title += f", {year}"
            title += ")"
        lines = [f"{title}:"]
        for item in holidays[:8]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('holiday_name')} on {item.get('date')} "
                    f"({item.get('location')})"
                )
        return "\n".join(lines)
    return "No holidays found for that country/year."


def _format_leave_history(output: dict[str, Any]) -> str:
    """Turn leave-history tool payloads into readable chat text."""
    history = output.get("leave_history") or []
    eid = output.get("employee_id") or "your account"
    year = output.get("year")
    total = output.get("total_days")
    title = f"Leave History — {eid}"
    if year is not None:
        title = f"{title} ({year})"
    if not history:
        return f"No leave history found for {eid}."

    header = f"{'Date':<14}{'Type':<12}{'Days':<8}{'Status'}"
    lines = [title, "", header]
    max_rows = 20
    shown = 0
    for item in history:
        if not isinstance(item, dict):
            continue
        if shown >= max_rows:
            remaining = sum(1 for row in history if isinstance(row, dict)) - shown
            if remaining > 0:
                lines.append(f"… and {remaining} more entries")
            break
        leave_type = str(
            item.get("type") or item.get("leave_type") or "LEAVE"
        ).replace("_", " ").title()
        days = item.get("days")
        status = str(item.get("status") or "").replace("_", " ").title()
        date = _format_leave_date(item.get("date"))
        day_label = f"{days:g}" if isinstance(days, float) else str(days)
        lines.append(f"{date:<14}{leave_type:<12}{day_label:<8}{status}")
        shown += 1
    if total is not None:
        total_label = f"{total:g}" if isinstance(total, float) else str(total)
        lines.append("")
        lines.append(f"Total leave used: {total_label} days")
    # Markdown code fence keeps columns aligned in Streamlit chat.
    return "```\n" + "\n".join(lines) + "\n```"


def _format_leave_date(value: Any) -> str:
    """Format ISO leave dates as ``Jan 03 2024`` when possible."""
    if value is None:
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    try:
        from datetime import date

        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return date.fromisoformat(text[:10]).strftime("%b %d %Y")
    except ValueError:
        pass
    return text


def _serialize_rag(shared_context: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a JSON-safe RAG snapshot from shared context."""
    response = shared_context.get("last_rag_response")
    if response is None:
        return None
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    return None


def contracts_from_state(state: AgentState) -> list[ToolContract]:
    """Rebuild ``ToolContract`` list from agent state payloads."""
    raw = state.get("tool_contracts") or []
    return [ToolContract.model_validate(item) for item in raw]


def executions_from_state(state: AgentState) -> list[ToolExecution]:
    """Rebuild ``ToolExecution`` list from agent state history."""
    raw = state.get("tool_execution_history") or []
    return [ToolExecution.model_validate(item) for item in raw]


__all__ = [
    "ToolRouter",
    "contracts_from_state",
    "executions_from_state",
]
