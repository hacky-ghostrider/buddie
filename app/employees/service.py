"""Employee service — verification boundary + protected data access.

Architecture:
    Employee JSON store → EmployeeService → employee tools → Agent

Protected methods require a verified employee id matching the session
context. Callers must not trust LLM-supplied employee_id overrides.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import date, timedelta
from typing import Any

from app.employees.exceptions import (
    EmployeeNotFoundError,
    EmployeeNotVerifiedError,
    EmployeeValidationError,
    EmployeeVerificationError,
)
from app.employees.generator import DEFAULT_AS_OF
from app.employees.models import (
    CompanyHoliday,
    EmployeeRecord,
    LeaveEligibilityResult,
    LeaveRequestResult,
    LeaveType,
    UpcomingLeave,
    VerifyEmployeeResponse,
    profile_dict,
)
from app.employees.store import EmployeeStore

logger = logging.getLogger(__name__)

_EMPLOYEE_ID_RE = re.compile(r"^E-\d{4}$")
_BARE_EMPLOYEE_DIGITS_RE = re.compile(r"^\d{4}$")
_VALID_LEAVE_TYPES: frozenset[str] = frozenset({"VACATION", "SICK", "PERSONAL"})

# Map holiday ``location`` codes / country aliases onto dataset values.
_COUNTRY_ALIASES: dict[str, str] = {
    "US": "US",
    "USA": "US",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "ALL": "All",
    "GLOBAL": "All",
    "COMPANY": "All",
}


def normalize_employee_id(value: str) -> str:
    """Normalize spacing/case for ids like ``E-1101``.

    Only exactly four bare digits are upgraded to ``E-####``. Shorter or
    longer numeric strings (e.g. ``123``) are left unchanged so they do not
    enter verification as malformed ids like ``E-123``.
    """
    cleaned = (value or "").strip().upper().replace(" ", "")
    if _BARE_EMPLOYEE_DIGITS_RE.match(cleaned):
        return f"E-{cleaned}"
    return cleaned


def normalize_leave_type(value: str) -> LeaveType:
    """Normalize leave-type aliases to dataset literals."""
    raw = (value or "").strip().upper().replace("-", " ").replace("_", " ")
    aliases = {
        "VACATION": "VACATION",
        "ANNUAL": "VACATION",
        "PTO": "VACATION",
        "PAID TIME OFF": "VACATION",
        "TIME OFF": "VACATION",
        "SICK": "SICK",
        "SICK LEAVE": "SICK",
        "PERSONAL": "PERSONAL",
        "PERSONAL LEAVE": "PERSONAL",
        "LEAVE": "VACATION",
    }
    mapped = aliases.get(raw)
    if mapped is None or mapped not in _VALID_LEAVE_TYPES:
        raise EmployeeValidationError(
            f"Invalid leave type '{value}'. "
            "Supported types: vacation, sick, personal."
        )
    return mapped  # type: ignore[return-value]


def _parse_iso_date(value: str, *, field: str) -> date:
    text = (value or "").strip()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise EmployeeValidationError(
            f"Invalid {field} '{value}'. Use YYYY-MM-DD."
        ) from exc


class EmployeeService:
    """Structured employee operations with verification gating."""

    def __init__(self, store: EmployeeStore | None = None) -> None:
        self._store = store or EmployeeStore()

    @property
    def store(self) -> EmployeeStore:
        """Underlying JSON store."""
        return self._store

    def ensure_ready(self) -> None:
        """Ensure the deterministic dataset is available."""
        self._store.ensure_seeded()

    def verify_employee(self, employee_id: str) -> VerifyEmployeeResponse:
        """Verify an employee id against the structured store.

        This is the only entry point that accepts a caller-supplied id.
        """
        eid = normalize_employee_id(employee_id)
        if not _EMPLOYEE_ID_RE.match(eid):
            raise EmployeeVerificationError(
                "That employee ID couldn't be verified.\n\n"
                "Employee IDs should follow the format E-1101.\n"
                "Please recheck and try again."
            )
        try:
            record = self._store.get_employee(eid)
        except EmployeeNotFoundError as exc:
            raise EmployeeVerificationError(
                "That employee ID couldn't be verified.\n\n"
                "Employee IDs should follow the format E-1101.\n"
                "Please recheck and try again."
            ) from exc

        logger.info("Employee verified: employee_id=%s", eid)
        return VerifyEmployeeResponse(
            verified=True,
            employee_id=record.employee_id,
            full_name=record.full_name,
            department=record.department,
            message="Verification successful",
            source="employee_store",
        )

    def require_verified(self, verified_employee_id: str | None) -> str:
        """Return a normalized verified id or raise."""
        if not verified_employee_id:
            raise EmployeeNotVerifiedError(
                "Before I access your employee information, I need to verify "
                "your employee ID.\n\n"
                "Please enter your employee ID, for example E-1101."
            )
        eid = normalize_employee_id(verified_employee_id)
        if not _EMPLOYEE_ID_RE.match(eid):
            raise EmployeeNotVerifiedError(
                "Employee verification required before accessing employee data."
            )
        # Ensure the verified id still exists (session cannot target others).
        self._store.get_employee(eid)
        return eid

    def _record(self, verified_employee_id: str | None) -> EmployeeRecord:
        eid = self.require_verified(verified_employee_id)
        return self._store.get_employee(eid)

    def get_employee_profile(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        """Profile fields for the verified employee only."""
        return profile_dict(self._record(verified_employee_id))

    def get_manager_information(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        """Manager-related fields sourced from the verified employee profile."""
        record = self._record(verified_employee_id)
        return {
            "employee_id": record.employee_id,
            "employee_name": record.full_name,
            "manager": record.manager,
            "department": record.department,
            "designation": record.designation,
            "location": record.location,
        }

    def get_leave_balance(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        record = self._record(verified_employee_id)
        return {
            "employee_id": record.employee_id,
            "leave_balance": record.leave_balance.model_dump(mode="json"),
        }

    def get_leave_history(
        self,
        verified_employee_id: str | None,
        *,
        year: int | None = None,
    ) -> dict[str, Any]:
        record = self._record(verified_employee_id)
        history = list(record.leave_history)
        if year is not None:
            history = [h for h in history if h.date.startswith(f"{year}-")]
        return {
            "employee_id": record.employee_id,
            "year": year,
            "leave_history": [h.model_dump(mode="json") for h in history],
            "total_days": sum(h.days for h in history if h.status == "APPROVED"),
        }

    def get_upcoming_leave(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        record = self._record(verified_employee_id)
        items = [u.model_dump(mode="json") for u in record.upcoming_leave]
        next_vacation = next(
            (
                u
                for u in record.upcoming_leave
                if u.leave_type == "VACATION"
                and u.status in {"APPROVED", "PENDING"}
            ),
            None,
        )
        pending = [u for u in record.upcoming_leave if u.status == "PENDING"]
        return {
            "employee_id": record.employee_id,
            "upcoming_leave": items,
            "next_vacation": (
                next_vacation.model_dump(mode="json") if next_vacation else None
            ),
            "pending_leave": [p.model_dump(mode="json") for p in pending],
        }

    def get_pending_actions(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        record = self._record(verified_employee_id)
        open_actions = [
            a
            for a in record.pending_actions
            if a.status in {"PENDING", "IN_PROGRESS", "OVERDUE"}
        ]
        return {
            "employee_id": record.employee_id,
            "pending_actions": [a.model_dump(mode="json") for a in open_actions],
            "count": len(open_actions),
        }

    def get_payroll_summary(
        self, verified_employee_id: str | None
    ) -> dict[str, Any]:
        """Payroll — verification already enforced by ``_record``."""
        record = self._record(verified_employee_id)
        return {
            "employee_id": record.employee_id,
            "payroll": record.payroll.model_dump(mode="json"),
        }

    def get_attendance_summary(
        self,
        verified_employee_id: str | None,
        *,
        year: int | None = None,
        month: int | None = None,
    ) -> dict[str, Any]:
        record = self._record(verified_employee_id)
        rows = list(record.attendance)
        if year is not None:
            rows = [r for r in rows if r.year == year]
        if month is not None:
            rows = [r for r in rows if r.month == month]
        return {
            "employee_id": record.employee_id,
            "year": year,
            "month": month,
            "attendance": [r.model_dump(mode="json") for r in rows],
        }

    def get_upcoming_holidays(
        self,
        *,
        as_of: date | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Company holidays — shared, no employee verification required."""
        today = as_of or DEFAULT_AS_OF
        holidays = sorted(self._store.holidays(), key=lambda h: h.date)
        upcoming = [
            h for h in holidays if date.fromisoformat(h.date) >= today
        ][: max(1, limit)]
        next_holiday: CompanyHoliday | None = upcoming[0] if upcoming else None
        return {
            "as_of_date": today.isoformat(),
            "upcoming_holidays": [h.model_dump(mode="json") for h in upcoming],
            "next_holiday": (
                next_holiday.model_dump(mode="json") if next_holiday else None
            ),
        }

    def get_holiday_calendar(
        self,
        *,
        country: str | None = None,
        year: int | None = None,
    ) -> dict[str, Any]:
        """Return holidays filtered by country/location and year.

        Uses only the seeded company holiday dataset. Unknown countries yield
        an empty list (no fabricated holidays).
        """
        normalized_country = self._normalize_country(country)
        target_year = year if year is not None else DEFAULT_AS_OF.year
        if not isinstance(target_year, int) or target_year < 2000 or target_year > 2100:
            raise EmployeeValidationError(
                f"Invalid year '{year}'. Provide a calendar year like 2026."
            )

        holidays = sorted(self._store.holidays(), key=lambda h: h.date)
        filtered: list[CompanyHoliday] = []
        for holiday in holidays:
            if not holiday.date.startswith(f"{target_year}-"):
                continue
            if normalized_country is None:
                filtered.append(holiday)
                continue
            if holiday.location in {normalized_country, "All"}:
                filtered.append(holiday)

        return {
            "country": normalized_country or "All",
            "year": target_year,
            "holidays": [h.model_dump(mode="json") for h in filtered],
            "count": len(filtered),
        }

    def check_leave_eligibility(
        self,
        verified_employee_id: str | None,
        *,
        leave_type: str,
        requested_days: float | int | str,
    ) -> dict[str, Any]:
        """Validate leave type / days against available balance and status."""
        record = self._record(verified_employee_id)
        normalized_type = normalize_leave_type(leave_type)
        days = self._parse_requested_days(requested_days)

        reasons: list[str] = []
        policy_notes: list[str] = []
        available = float(getattr(record.leave_balance, normalized_type.lower()))

        if record.employment_status != "ACTIVE":
            reasons.append(
                f"Employment status is {record.employment_status}; "
                "only ACTIVE employees may request leave."
            )

        if days <= 0:
            reasons.append("Requested days must be greater than zero.")

        if days > available:
            reasons.append(
                f"Insufficient {normalized_type.lower()} balance: "
                f"requested {days:g} day(s), available {available:g}."
            )

        # Lightweight business rule from existing data — no invented policy.
        if normalized_type == "VACATION" and days > 10:
            policy_notes.append(
                "Requests over 10 vacation days typically need manager approval."
            )

        eligible = not reasons
        if eligible:
            reasons.append(
                f"Eligible: {days:g} {normalized_type.lower()} day(s) "
                f"within available balance of {available:g}."
            )

        result = LeaveEligibilityResult(
            employee_id=record.employee_id,
            leave_type=normalized_type,
            requested_days=days,
            available_days=available,
            eligible=eligible,
            reasons=reasons,
            policy_notes=policy_notes,
        )
        return result.model_dump(mode="json")

    def create_leave_request(
        self,
        verified_employee_id: str | None,
        *,
        leave_type: str,
        start_date: str,
        end_date: str,
        reason: str,
        confirmed: bool = False,
    ) -> dict[str, Any]:
        """Create a PENDING leave request after explicit confirmation.

        The write is refused unless ``confirmed`` is True. Callers (tools /
        planner) must obtain human confirmation before setting that flag.
        """
        if not confirmed:
            raise EmployeeValidationError(
                "Leave request was not created. Explicit user confirmation "
                "is required before write actions."
            )

        record = self._record(verified_employee_id)
        normalized_type = normalize_leave_type(leave_type)
        start = _parse_iso_date(start_date, field="start_date")
        end = _parse_iso_date(end_date, field="end_date")
        if end < start:
            raise EmployeeValidationError(
                "end_date must be on or after start_date."
            )
        reason_text = (reason or "").strip() or "Employee leave request"
        number_of_days = float((end - start).days + 1)
        if number_of_days <= 0:
            raise EmployeeValidationError(
                "Leave window must cover at least one day."
            )

        eligibility = self.check_leave_eligibility(
            record.employee_id,
            leave_type=normalized_type,
            requested_days=number_of_days,
        )
        if not eligibility["eligible"]:
            raise EmployeeValidationError(
                "; ".join(eligibility.get("reasons") or ["Not eligible"])
            )

        request_id = f"LR-{uuid.uuid4().hex[:8].upper()}"
        upcoming = UpcomingLeave(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            leave_type=normalized_type,
            status="PENDING",
            number_of_days=number_of_days,
        )
        updated = record.model_copy(
            update={"upcoming_leave": list(record.upcoming_leave) + [upcoming]}
        )
        self._store.update_employee(updated)

        result = LeaveRequestResult(
            created=True,
            request_id=request_id,
            employee_id=record.employee_id,
            leave_type=normalized_type,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            reason=reason_text,
            number_of_days=number_of_days,
            status="PENDING",
            message=(
                f"Leave request {request_id} created as PENDING "
                f"({normalized_type.lower()} {start.isoformat()} → "
                f"{end.isoformat()})."
            ),
        )
        logger.info(
            "Leave request created: employee_id=%s request_id=%s",
            record.employee_id,
            request_id,
        )
        return result.model_dump(mode="json")

    @staticmethod
    def propose_leave_window(
        *,
        requested_days: float,
        relative_hint: str | None = None,
        as_of: date | None = None,
    ) -> tuple[str, str]:
        """Propose start/end dates from a relative phrase using demo ``as_of``."""
        today = as_of or DEFAULT_AS_OF
        days = max(1, int(round(float(requested_days))))
        hint = (relative_hint or "").lower()
        if "next month" in hint:
            year = today.year + (1 if today.month == 12 else 0)
            month = 1 if today.month == 12 else today.month + 1
            start = date(year, month, 1)
        elif "next week" in hint:
            start = today + timedelta(days=(7 - today.weekday()) or 7)
        else:
            start = today + timedelta(days=1)
        end = start + timedelta(days=days - 1)
        return start.isoformat(), end.isoformat()

    @staticmethod
    def _normalize_country(country: str | None) -> str | None:
        if country is None or str(country).strip() == "":
            return None
        key = str(country).strip().upper()
        if key in _COUNTRY_ALIASES:
            return _COUNTRY_ALIASES[key]
        # Accept exact dataset location codes as-is (e.g. already ``US``).
        return key.title() if key not in {"US", "ALL"} else key

    @staticmethod
    def _parse_requested_days(value: float | int | str) -> float:
        try:
            days = float(value)
        except (TypeError, ValueError) as exc:
            raise EmployeeValidationError(
                f"Invalid requested_days '{value}'. Provide a positive number."
            ) from exc
        if days <= 0:
            raise EmployeeValidationError(
                "requested_days must be greater than zero."
            )
        if days > 365:
            raise EmployeeValidationError(
                "requested_days is unreasonably large (max 365)."
            )
        return days


__all__ = [
    "EmployeeService",
    "normalize_employee_id",
    "normalize_leave_type",
]
