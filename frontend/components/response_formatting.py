"""Presentation-layer formatting for assistant answers.

Formats structured tool payloads for chat display without changing tool
response contracts. Prefer backend-provided ``final_answer`` when it is
already human-readable; reformat only when a raw structured dump slipped
through.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from typing import Any

_RAW_DICT_MARKERS = (
    "'leave_history'",
    '"leave_history"',
    "'pending_actions'",
    '"pending_actions"',
    "'leave_balance'",
    '"leave_balance"',
)


def looks_like_raw_structured_dump(answer: str) -> bool:
    """Detect leftover Python/JSON dict dumps that should not reach chat."""
    text = (answer or "").strip()
    if not text:
        return False
    if not (text.startswith("{") or text.startswith("{'") or text.startswith('{"')):
        return False
    return any(marker in text for marker in _RAW_DICT_MARKERS)


def format_assistant_answer(mode: str, payload: dict[str, Any]) -> str:
    """Pick a user-facing answer, reformatting structured tool dumps when needed."""
    if mode == "RAG":
        return str(payload.get("answer") or "")

    answer = str(payload.get("final_answer") or "")
    hybrid = "policy:" in answer.lower()

    # Prefer structured presentation for known employee tools unless this is a
    # hybrid leave+policy answer that already includes both parts.
    if not hybrid:
        tools = payload.get("tool_executions") or []
        for tool in reversed(tools):
            if not isinstance(tool, dict):
                continue
            status = str(tool.get("status") or "").lower()
            if status and status not in {"success", "ok", "succeeded", "completed"}:
                if tool.get("error"):
                    continue
            output = tool.get("output")
            if not isinstance(output, dict) or not output:
                continue
            tool_name = str(tool.get("tool_name") or "")
            if tool_name and not (
                tool_name.startswith("get_") or tool_name == "verify_employee"
            ):
                continue
            formatted = format_tool_output(tool_name, output)
            if formatted:
                return formatted

    if answer and not looks_like_raw_structured_dump(answer):
        return answer

    # Last resort: parse a raw dict string answer.
    parsed = _parse_dict_literal(answer)
    if isinstance(parsed, dict):
        if "leave_history" in parsed:
            return format_leave_history(parsed)
        if "leave_balance" in parsed:
            return format_leave_balance(parsed)
        if "pending_actions" in parsed:
            return format_pending_actions(parsed)
        if "upcoming_holidays" in parsed or "next_holiday" in parsed:
            return format_upcoming_holidays(parsed)
        if "employee_id" in parsed and "full_name" in parsed:
            return format_employee_profile(parsed)

    if looks_like_raw_structured_dump(answer):
        return (
            "I found the information, but had trouble formatting it. "
            "Please try asking again."
        )
    return answer


def format_tool_output(tool_name: str, output: dict[str, Any]) -> str | None:
    """Format known employee tool payloads for chat presentation."""
    if tool_name == "get_leave_history" or "leave_history" in output:
        return format_leave_history(output)
    if tool_name == "get_leave_balance" or "leave_balance" in output:
        return format_leave_balance(output)
    if tool_name == "get_pending_actions" or "pending_actions" in output:
        return format_pending_actions(output)
    if tool_name == "get_upcoming_holidays" or "upcoming_holidays" in output:
        return format_upcoming_holidays(output)
    if tool_name == "get_employee_profile" or (
        "full_name" in output and "employee_id" in output and "designation" in output
    ):
        return format_employee_profile(output)
    return None


def format_leave_balance(output: dict[str, Any]) -> str:
    """Render leave-balance tool payloads as a clean readable block."""
    lb = output.get("leave_balance") or {}
    vacation = lb.get("vacation")
    sick = lb.get("sick")
    personal = lb.get("personal")
    lines = [
        "Leave Balance",
        "",
        f"{'Vacation':<12}{vacation} days",
        f"{'Sick':<12}{sick} days",
        f"{'Personal':<12}{personal} days",
    ]
    return "\n".join(lines)


def format_upcoming_holidays(output: dict[str, Any]) -> str:
    """Render holiday tool payloads as a readable list (never fabricated)."""
    nxt = output.get("next_holiday")
    holidays = output.get("upcoming_holidays") or []
    if isinstance(nxt, dict):
        lines = [
            "Upcoming Holidays",
            "",
            f"Next: {nxt.get('holiday_name')} on {nxt.get('date')} "
            f"({nxt.get('location')})",
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
        lines = ["Upcoming Holidays", ""]
        for item in holidays[:5]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('holiday_name')} on {item.get('date')} "
                    f"({item.get('location')})"
                )
        return "\n".join(lines)
    return "No upcoming company holidays found."


def format_employee_profile(output: dict[str, Any]) -> str:
    """Render employee-profile tool payloads as concise chat text."""
    name = output.get("full_name") or "Employee"
    eid = output.get("employee_id") or "—"
    designation = output.get("designation") or "—"
    department = output.get("department") or "—"
    manager = output.get("manager") or "—"
    return (
        f"Employee Profile\n\n"
        f"{name} ({eid})\n"
        f"{designation} · {department}\n"
        f"Manager: {manager}"
    )


def format_pending_actions(output: dict[str, Any]) -> str:
    """Render pending-actions tool payloads as a readable list."""
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


def format_leave_history(output: dict[str, Any], *, max_rows: int = 20) -> str:
    """Render leave-history payloads as an aligned, scrollable table block."""
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
    shown = 0
    typed_rows = [row for row in history if isinstance(row, dict)]
    for item in typed_rows:
        if shown >= max_rows:
            remaining = len(typed_rows) - shown
            if remaining > 0:
                lines.append(f"… and {remaining} more entries")
            break
        leave_type = str(
            item.get("type") or item.get("leave_type") or "LEAVE"
        ).replace("_", " ").title()
        days = item.get("days")
        status = str(item.get("status") or "").replace("_", " ").title()
        day_label = f"{days:g}" if isinstance(days, float) else str(days)
        lines.append(
            f"{_format_leave_date(item.get('date')):<14}"
            f"{leave_type:<12}{day_label:<8}{status}"
        )
        shown += 1
    if total is not None:
        total_label = f"{total:g}" if isinstance(total, float) else str(total)
        lines.append("")
        lines.append(f"Total leave used: {total_label} days")
    return "```\n" + "\n".join(lines) + "\n```"


def _format_leave_date(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    try:
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return date.fromisoformat(text[:10]).strftime("%b %d %Y")
    except ValueError:
        pass
    return text


def _parse_dict_literal(text: str) -> Any:
    cleaned = (text or "").strip()
    if not cleaned.startswith("{"):
        return None
    try:
        return ast.literal_eval(cleaned)
    except (SyntaxError, ValueError, MemoryError):
        pass
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


__all__ = [
    "looks_like_raw_structured_dump",
    "format_assistant_answer",
    "format_tool_output",
    "format_leave_history",
    "format_leave_balance",
    "format_pending_actions",
    "format_upcoming_holidays",
    "format_employee_profile",
]
