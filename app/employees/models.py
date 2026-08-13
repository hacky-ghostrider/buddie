"""Pydantic models for the structured employee HR dataset.

Employee operational data lives here — not in the RAG vector store.
Policies / handbooks remain RAG documents; profiles, leave, payroll,
attendance, and pending actions are served by employee tools.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LeaveType = Literal["VACATION", "SICK", "PERSONAL"]
LeaveStatus = Literal["APPROVED", "PENDING", "REJECTED", "CANCELLED"]
EmploymentStatus = Literal["ACTIVE", "ON_LEAVE", "TERMINATED"]
EmploymentType = Literal["FULL_TIME", "PART_TIME", "CONTRACT"]
WorkMode = Literal["ONSITE", "HYBRID", "REMOTE"]
ActionStatus = Literal["PENDING", "IN_PROGRESS", "COMPLETED", "OVERDUE"]
ActionPriority = Literal["LOW", "MEDIUM", "HIGH"]
PayrollStatus = Literal["CURRENT", "PROCESSING", "ON_HOLD"]


class LeaveBalance(BaseModel):
    """Current leave balances (days remaining)."""

    model_config = ConfigDict(extra="forbid")

    vacation: int = Field(ge=0)
    sick: int = Field(ge=0)
    personal: int = Field(ge=0)


class LeaveHistoryRecord(BaseModel):
    """One historical leave usage record."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(description="ISO date YYYY-MM-DD (leave start)")
    type: LeaveType
    days: float = Field(gt=0)
    status: LeaveStatus
    reason: str


class UpcomingLeave(BaseModel):
    """Approved or pending future leave window."""

    model_config = ConfigDict(extra="forbid")

    start_date: str
    end_date: str
    leave_type: LeaveType
    status: LeaveStatus
    number_of_days: float = Field(gt=0)


class PayrollSummary(BaseModel):
    """Fictional payroll snapshot — gated behind verify_employee."""

    model_config = ConfigDict(extra="forbid")

    monthly_gross: float = Field(gt=0)
    monthly_net: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    last_pay_date: str
    next_pay_date: str
    payroll_status: PayrollStatus
    pending_payroll_action: str | None = None


class PendingAction(BaseModel):
    """One HR / compliance pending task."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    description: str
    due_date: str
    status: ActionStatus
    priority: ActionPriority


class AttendanceMonth(BaseModel):
    """Monthly attendance rollup."""

    model_config = ConfigDict(extra="forbid")

    year: int
    month: int = Field(ge=1, le=12)
    working_days: int = Field(ge=0)
    days_present: int = Field(ge=0)
    days_absent: int = Field(ge=0)
    leave_days: float = Field(ge=0)
    late_days: int = Field(ge=0)


class EmployeeProfile(BaseModel):
    """Core employee profile fields (non-sensitive identity + org)."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str
    full_name: str
    department: str
    designation: str
    manager: str
    location: str
    joining_date: str
    employment_status: EmploymentStatus
    employment_type: EmploymentType
    work_mode: WorkMode

    @field_validator("employee_id")
    @classmethod
    def employee_id_format(cls, value: str) -> str:
        """Normalize and require ``E-####`` ids."""
        cleaned = value.strip().upper()
        if not cleaned.startswith("E-") or len(cleaned) < 4:
            raise ValueError("employee_id must look like E-1101")
        return cleaned


class EmployeeRecord(EmployeeProfile):
    """Full operational record for one employee."""

    leave_balance: LeaveBalance
    leave_history: list[LeaveHistoryRecord] = Field(default_factory=list)
    upcoming_leave: list[UpcomingLeave] = Field(default_factory=list)
    payroll: PayrollSummary
    pending_actions: list[PendingAction] = Field(default_factory=list)
    attendance: list[AttendanceMonth] = Field(default_factory=list)


class CompanyHoliday(BaseModel):
    """Shared company holiday calendar entry."""

    model_config = ConfigDict(extra="forbid")

    holiday_name: str
    date: str
    location: str = Field(
        default="All",
        description="Region / office applicability (All = company-wide)",
    )


class EmployeeDataset(BaseModel):
    """Top-level deterministic employee dataset document."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    seed: int
    as_of_date: str
    employee_count: int
    employees: list[EmployeeRecord]
    holidays: list[CompanyHoliday]

    def by_id(self) -> dict[str, EmployeeRecord]:
        """Index employees by id."""
        return {emp.employee_id: emp for emp in self.employees}


class VerifyEmployeeRequest(BaseModel):
    """HTTP body for ``POST /employees/verify``."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str

    @field_validator("employee_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        """Strip / uppercase employee ids."""
        cleaned = (value or "").strip().upper().replace(" ", "")
        if cleaned and not cleaned.startswith("E-") and cleaned[0].isdigit():
            cleaned = f"E-{cleaned}"
        return cleaned


class VerifyEmployeeResponse(BaseModel):
    """Successful or failed verification payload."""

    model_config = ConfigDict(extra="forbid")

    verified: bool
    employee_id: str | None = None
    full_name: str | None = None
    department: str | None = None
    message: str | None = None
    source: str = "employee_store"


class LeaveEligibilityResult(BaseModel):
    """Structured result from ``check_leave_eligibility``."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str
    leave_type: LeaveType
    requested_days: float = Field(gt=0)
    available_days: float = Field(ge=0)
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    policy_notes: list[str] = Field(default_factory=list)


class LeaveRequestResult(BaseModel):
    """Structured result from ``create_leave_request``."""

    model_config = ConfigDict(extra="forbid")

    created: bool
    request_id: str | None = None
    employee_id: str
    leave_type: LeaveType
    start_date: str
    end_date: str
    reason: str
    number_of_days: float = Field(gt=0)
    status: LeaveStatus = "PENDING"
    message: str


def profile_dict(record: EmployeeRecord) -> dict[str, Any]:
    """Serialize profile-only fields for tool output.

    ``EmployeeRecord`` carries operational fields that ``EmployeeProfile``
    forbids, so dump only profile keys before validation.
    """
    profile_keys = set(EmployeeProfile.model_fields.keys())
    return EmployeeProfile.model_validate(
        record.model_dump(include=profile_keys)
    ).model_dump(mode="json")


__all__ = [
    "LeaveBalance",
    "LeaveHistoryRecord",
    "UpcomingLeave",
    "PayrollSummary",
    "PendingAction",
    "AttendanceMonth",
    "EmployeeProfile",
    "EmployeeRecord",
    "CompanyHoliday",
    "EmployeeDataset",
    "VerifyEmployeeRequest",
    "VerifyEmployeeResponse",
    "LeaveEligibilityResult",
    "LeaveRequestResult",
    "profile_dict",
]
