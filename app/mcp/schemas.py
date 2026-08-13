"""Typed MCP tool input / output contracts for Buddie tools.

Descriptions are written for MCP clients / LLMs. They intentionally omit
internal implementation details and secrets.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class McpToolResult(BaseModel):
    """Standard envelope returned by Buddie MCP tools."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = Field(description="Whether the tool completed successfully")
    tool: str = Field(description="Tool name that produced this result")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured tool payload from the existing Buddie services",
    )
    error: str | None = Field(
        default=None,
        description="Safe error message when ok is false",
    )
    error_code: str | None = Field(
        default=None,
        description="Stable error code for clients (not a stack trace)",
    )


# ---------------------------------------------------------------------------
# Input schemas (documented for discovery; FastMCP also derives from fn sigs)
# ---------------------------------------------------------------------------


class EmptyProtectedInput(BaseModel):
    """No user-supplied employee id — uses verified session context only."""

    model_config = ConfigDict(extra="forbid")


class LeaveHistoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year: int | None = Field(
        default=None,
        description="Optional calendar year filter for leave history entries",
    )


class HolidayCalendarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    country: str | None = Field(
        default=None,
        description="Optional country / location filter (e.g. US, IN). "
        "Omit to include company-wide holidays.",
    )
    year: int | None = Field(
        default=None,
        description="Optional calendar year. Defaults to the current year.",
    )


class LeaveEligibilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leave_type: str = Field(
        default="VACATION",
        description="Leave category to check (VACATION, SICK, PERSONAL, etc.)",
    )
    requested_days: float = Field(
        default=1,
        gt=0,
        description="Number of leave days the employee wants to take",
    )


class SearchCompanyPolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        description="Natural-language policy question to search in company docs",
    )


class CreateLeaveRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leave_type: str = Field(
        default="VACATION",
        description="Leave category for the request",
    )
    start_date: str = Field(
        description="Inclusive start date in ISO format YYYY-MM-DD",
    )
    end_date: str = Field(
        description="Inclusive end date in ISO format YYYY-MM-DD",
    )
    reason: str = Field(
        default="Employee leave request",
        description="Short reason shown on the leave request",
    )
    confirmed: bool = Field(
        default=False,
        description=(
            "Must be true only after the employee explicitly confirms the write. "
            "Never set this without human confirmation."
        ),
    )


EXPECTED_MCP_TOOL_NAMES: tuple[str, ...] = (
    "get_leave_balance",
    "get_leave_history",
    "get_employee_profile",
    "get_manager_information",
    "get_holiday_calendar",
    "check_leave_eligibility",
    "search_company_policy",
    "create_leave_request",
)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_leave_balance": (
        "Retrieve the authenticated employee's current leave balances "
        "(vacation, sick, personal, etc.). Requires a verified employee "
        "context from the Buddie session. Do not use for arbitrary employee "
        "IDs — only the verified caller may be queried."
    ),
    "get_leave_history": (
        "Retrieve leave history for the authenticated employee. Optional year "
        "filter. Requires verified employee context; never accepts another "
        "employee's id from the user."
    ),
    "get_employee_profile": (
        "Retrieve the authenticated employee's profile (name, department, "
        "title, status). Requires verified employee context. Do not use to "
        "look up other employees."
    ),
    "get_manager_information": (
        "Retrieve manager information for the authenticated employee. "
        "Requires verified employee context."
    ),
    "get_holiday_calendar": (
        "Retrieve the company holiday calendar, optionally filtered by "
        "country and year. Shared calendar data — does not expose private "
        "employee records."
    ),
    "check_leave_eligibility": (
        "Check whether the authenticated employee can take the requested "
        "number of days for a leave type against available balance and "
        "employment status. Requires verified employee context."
    ),
    "search_company_policy": (
        "Search company policy / handbook documents for an answer to a "
        "policy question (e.g. carry-forward rules). Use when policy text "
        "is needed in addition to personal leave data."
    ),
    "create_leave_request": (
        "WRITE operation: create a pending leave request for the authenticated "
        "employee. Requires verified employee context AND explicit human "
        "confirmation (confirmed=true). Never bypass confirmation."
    ),
}


__all__ = [
    "McpToolResult",
    "EmptyProtectedInput",
    "LeaveHistoryInput",
    "HolidayCalendarInput",
    "LeaveEligibilityInput",
    "SearchCompanyPolicyInput",
    "CreateLeaveRequestInput",
    "EXPECTED_MCP_TOOL_NAMES",
    "TOOL_DESCRIPTIONS",
]
