"""Employee agent tools — structured HR data via verified session context.

Protected tools read ``verified_employee_id`` from the shared tool context
(populated from AgentRequest metadata after UI/API verification). LLM-supplied
``employee_id`` arguments on protected tools are ignored.

Write tools (``create_leave_request``) additionally require an explicit
``confirmed=True`` flag set only after human confirmation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from app.employees.exceptions import (
    EmployeeError,
    EmployeeNotVerifiedError,
    EmployeeValidationError,
    EmployeeVerificationError,
)
from app.employees.service import EmployeeService
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionMetrics,
    ToolExecutionStatus,
)

logger = logging.getLogger(__name__)

PROTECTED_EMPLOYEE_TOOLS = frozenset(
    {
        "get_employee_profile",
        "get_manager_information",
        "get_leave_balance",
        "get_leave_history",
        "get_upcoming_leave",
        "get_pending_actions",
        "get_payroll_summary",
        "get_attendance_summary",
        "check_leave_eligibility",
        "create_leave_request",
    }
)

WRITE_EMPLOYEE_TOOLS = frozenset({"create_leave_request"})


def verified_employee_id_from_context(
    context: dict[str, Any] | None,
) -> str | None:
    """Extract verified employee id from router / agent metadata context."""
    if not context:
        return None
    for key in ("verified_employee_id", "employee_id"):
        value = context.get(key)
        if value:
            return str(value).strip().upper()
    metadata = context.get("metadata")
    if isinstance(metadata, dict):
        for key in ("verified_employee_id", "employee_id"):
            value = metadata.get(key)
            if value:
                return str(value).strip().upper()
    return None


def _execution(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    order: int,
    correlation_id: str | None,
    started_at: datetime,
    started: float,
    output: dict[str, Any] | None = None,
    error: str | None = None,
) -> ToolExecution:
    latency_ms = (time.perf_counter() - started) * 1000.0
    finished_at = datetime.now(timezone.utc)
    status = (
        ToolExecutionStatus.FAILED if error else ToolExecutionStatus.SUCCESS
    )
    metrics = ToolExecutionMetrics.from_latency(
        execution_time_ms=latency_ms,
        status=status,
        failure_reason=error,
        started_at=started_at,
        finished_at=finished_at,
    )
    if error:
        logger.warning(
            "%s failed: correlation_id=%s error=%s",
            tool_name,
            correlation_id,
            error,
        )
    else:
        logger.info(
            "%s completed: correlation_id=%s",
            tool_name,
            correlation_id,
        )
    return ToolExecution(
        tool_name=tool_name,
        arguments=arguments,
        output=output or {},
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=latency_ms,
        status=status,
        error=error,
        order=order,
        metrics=metrics,
        trace_metadata={"tool": tool_name, "domain": "employee"},
    )


class _EmployeeToolBase:
    """Shared helpers for employee tools."""

    name: str = ""

    def __init__(self, service: EmployeeService) -> None:
        self._service = service

    def _run(
        self,
        arguments: dict[str, Any],
        *,
        order: int,
        correlation_id: str | None,
        context: dict[str, Any] | None,
        call: Callable[[], dict[str, Any]],
        recorded_arguments: dict[str, Any] | None = None,
    ) -> ToolExecution:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        args = recorded_arguments if recorded_arguments is not None else dict(arguments)
        try:
            output = call()
            # Stash a human-readable answer fragment for the router.
            if context is not None and "summary" not in context:
                context["employee_tool_result"] = output
            return _execution(
                tool_name=self.name,
                arguments=args,
                order=order,
                correlation_id=correlation_id,
                started_at=started_at,
                started=started,
                output=output,
            )
        except (EmployeeNotVerifiedError, EmployeeVerificationError) as exc:
            return _execution(
                tool_name=self.name,
                arguments=args,
                order=order,
                correlation_id=correlation_id,
                started_at=started_at,
                started=started,
                error=str(exc),
            )
        except EmployeeError as exc:
            return _execution(
                tool_name=self.name,
                arguments=args,
                order=order,
                correlation_id=correlation_id,
                started_at=started_at,
                started=started,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return _execution(
                tool_name=self.name,
                arguments=args,
                order=order,
                correlation_id=correlation_id,
                started_at=started_at,
                started=started,
                error=str(exc),
            )


class VerifyEmployeeTool(_EmployeeToolBase):
    """Public tool: verify an employee id and bind session context."""

    name = "verify_employee"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        employee_id = str(
            arguments.get("employee_id")
            or arguments.get("id")
            or ""
        )

        def call() -> dict[str, Any]:
            result = self._service.verify_employee(employee_id)
            payload = result.model_dump(mode="json")
            if context is not None and result.verified and result.employee_id:
                context["verified_employee_id"] = result.employee_id
                context["employee_id"] = result.employee_id
            return payload

        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=call,
            recorded_arguments={"employee_id": employee_id.strip().upper()},
        )


class _ProtectedEmployeeTool(_EmployeeToolBase):
    """Base for tools that only operate on the verified session employee."""

    def _verified_id(
        self, arguments: dict[str, Any], context: dict[str, Any] | None
    ) -> str | None:
        # Intentionally ignore arguments["employee_id"] to prevent LLM spoofing.
        del arguments
        return verified_employee_id_from_context(context)


class GetEmployeeProfileTool(_ProtectedEmployeeTool):
    name = "get_employee_profile"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_employee_profile(verified),
            recorded_arguments={},
        )


class GetManagerInformationTool(_ProtectedEmployeeTool):
    name = "get_manager_information"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_manager_information(verified),
            recorded_arguments={},
        )


class GetLeaveBalanceTool(_ProtectedEmployeeTool):
    name = "get_leave_balance"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_leave_balance(verified),
            recorded_arguments={},
        )


class GetLeaveHistoryTool(_ProtectedEmployeeTool):
    name = "get_leave_history"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        year_raw = arguments.get("year")
        year = int(year_raw) if year_raw not in (None, "") else None
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_leave_history(verified, year=year),
            recorded_arguments={"year": year},
        )


class GetUpcomingLeaveTool(_ProtectedEmployeeTool):
    name = "get_upcoming_leave"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_upcoming_leave(verified),
            recorded_arguments={},
        )


class GetPendingActionsTool(_ProtectedEmployeeTool):
    name = "get_pending_actions"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_pending_actions(verified),
            recorded_arguments={},
        )


class GetPayrollSummaryTool(_ProtectedEmployeeTool):
    name = "get_payroll_summary"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_payroll_summary(verified),
            recorded_arguments={},
        )


class GetAttendanceSummaryTool(_ProtectedEmployeeTool):
    name = "get_attendance_summary"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        year_raw = arguments.get("year")
        month_raw = arguments.get("month")
        year = int(year_raw) if year_raw not in (None, "") else None
        month = int(month_raw) if month_raw not in (None, "") else None
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.get_attendance_summary(
                verified, year=year, month=month
            ),
            recorded_arguments={"year": year, "month": month},
        )


class CheckLeaveEligibilityTool(_ProtectedEmployeeTool):
    name = "check_leave_eligibility"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        leave_type = str(arguments.get("leave_type") or "VACATION")
        requested_days = arguments.get("requested_days", 1)
        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=lambda: self._service.check_leave_eligibility(
                verified,
                leave_type=leave_type,
                requested_days=requested_days,  # type: ignore[arg-type]
            ),
            recorded_arguments={
                "leave_type": leave_type,
                "requested_days": requested_days,
            },
        )


class CreateLeaveRequestTool(_ProtectedEmployeeTool):
    """WRITE tool — refuses unless ``confirmed`` is true."""

    name = "create_leave_request"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        verified = self._verified_id(arguments, context)
        leave_type = str(arguments.get("leave_type") or "VACATION")
        start_date = str(arguments.get("start_date") or "")
        end_date = str(arguments.get("end_date") or "")
        reason = str(arguments.get("reason") or "Employee leave request")
        confirmed = bool(arguments.get("confirmed"))
        # Context flag may be set by the planner after human confirmation.
        if not confirmed and context is not None:
            confirmed = bool(context.get("leave_request_confirmed"))

        recorded = {
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
            "reason": reason,
            "confirmed": confirmed,
        }

        def call() -> dict[str, Any]:
            if not confirmed:
                raise EmployeeValidationError(
                    "Leave request was not created. Explicit user confirmation "
                    "is required before write actions."
                )
            return self._service.create_leave_request(
                verified,
                leave_type=leave_type,
                start_date=start_date,
                end_date=end_date,
                reason=reason,
                confirmed=True,
            )

        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=call,
            recorded_arguments=recorded,
        )


class GetUpcomingHolidaysTool(_EmployeeToolBase):
    """Shared holiday calendar — no employee verification required."""

    name = "get_upcoming_holidays"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        limit_raw = arguments.get("limit", 5)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError):
            limit = 5

        def call() -> dict[str, Any]:
            return self._service.get_upcoming_holidays(limit=limit)

        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=call,
            recorded_arguments={"limit": limit},
        )


class GetHolidayCalendarTool(_EmployeeToolBase):
    """Holiday calendar by country/year — no employee verification required."""

    name = "get_holiday_calendar"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        country = arguments.get("country")
        year_raw = arguments.get("year")
        year = int(year_raw) if year_raw not in (None, "") else None
        country_str = str(country).strip() if country not in (None, "") else None

        def call() -> dict[str, Any]:
            return self._service.get_holiday_calendar(
                country=country_str,
                year=year,
            )

        return self._run(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
            call=call,
            recorded_arguments={"country": country_str, "year": year},
        )


def build_employee_tools(
    service: EmployeeService | None = None,
) -> list[Any]:
    """Instantiate the full employee tool set."""
    svc = service or EmployeeService()
    svc.ensure_ready()
    return [
        VerifyEmployeeTool(svc),
        GetEmployeeProfileTool(svc),
        GetManagerInformationTool(svc),
        GetLeaveBalanceTool(svc),
        GetLeaveHistoryTool(svc),
        GetUpcomingLeaveTool(svc),
        GetPendingActionsTool(svc),
        GetPayrollSummaryTool(svc),
        GetAttendanceSummaryTool(svc),
        CheckLeaveEligibilityTool(svc),
        CreateLeaveRequestTool(svc),
        GetUpcomingHolidaysTool(svc),
        GetHolidayCalendarTool(svc),
    ]


__all__ = [
    "PROTECTED_EMPLOYEE_TOOLS",
    "WRITE_EMPLOYEE_TOOLS",
    "VerifyEmployeeTool",
    "GetEmployeeProfileTool",
    "GetManagerInformationTool",
    "GetLeaveBalanceTool",
    "GetLeaveHistoryTool",
    "GetUpcomingLeaveTool",
    "GetPendingActionsTool",
    "GetPayrollSummaryTool",
    "GetAttendanceSummaryTool",
    "CheckLeaveEligibilityTool",
    "CreateLeaveRequestTool",
    "GetUpcomingHolidaysTool",
    "GetHolidayCalendarTool",
    "build_employee_tools",
    "verified_employee_id_from_context",
]
