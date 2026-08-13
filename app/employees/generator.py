"""Deterministic fictional employee dataset generator.

Running the generator with the same seed always yields the same records.
No real employee information is used.
"""

from __future__ import annotations

import hashlib
from calendar import monthrange
from datetime import date, timedelta
from typing import Any

from app.employees.models import (
    AttendanceMonth,
    CompanyHoliday,
    EmployeeDataset,
    EmployeeRecord,
    LeaveBalance,
    LeaveHistoryRecord,
    UpcomingLeave,
    PayrollSummary,
    PendingAction,
)

# Stable demo constants — predictable answers in interviews / tests.
DEFAULT_SEED = 20260808
DEFAULT_AS_OF = date(2026, 8, 8)
DEFAULT_EMPLOYEE_COUNT = 30
FIRST_EMPLOYEE_NUM = 1101

DEPARTMENTS: list[tuple[str, str]] = [
    ("Engineering", "Software Engineer"),
    ("Quality Engineering", "QA Engineer"),
    ("Product", "Product Manager"),
    ("Data", "Data Analyst"),
    ("HR", "HR Specialist"),
    ("Finance", "Financial Analyst"),
    ("IT", "IT Support Specialist"),
    ("Customer Support", "Support Associate"),
    ("Sales", "Account Executive"),
    ("Operations", "Operations Coordinator"),
]

SENIOR_TITLES = {
    "Engineering": "Senior Software Engineer",
    "Quality Engineering": "Senior QA Engineer",
    "Product": "Senior Product Manager",
    "Data": "Senior Data Analyst",
    "HR": "HR Business Partner",
    "Finance": "Senior Financial Analyst",
    "IT": "IT Systems Administrator",
    "Customer Support": "Support Team Lead",
    "Sales": "Senior Account Executive",
    "Operations": "Operations Manager",
}

FIRST_NAMES = [
    "Avery", "Jordan", "Riley", "Casey", "Morgan", "Quinn", "Taylor", "Reese",
    "Cameron", "Harper", "Drew", "Skyler", "Parker", "Rowan", "Sage", "Blake",
    "Finley", "Hayden", "Logan", "Emery", "Kendall", "Peyton", "Jamie", "Alex",
    "Charlie", "Dakota", "Elliot", "Frankie", "Gray", "Hunter",
]

LAST_NAMES = [
    "Nguyen", "Patel", "Brooks", "Kim", "Garcia", "Okafor", "Singh", "Chen",
    "Martinez", "Ali", "Novak", "Berg", "Diaz", "Walsh", "Khan", "Ito",
    "Silva", "Cohen", "Murphy", "Park", "Rossi", "Anders", "Torres", "Lee",
    "Hughes", "Bennett", "Cruz", "Shah", "Adams", "Vogel",
]

LOCATIONS = [
    "Austin", "Seattle", "Chicago", "Remote-US", "Boston",
    "Denver", "Atlanta", "Remote-EU", "New York", "San Francisco",
]

MANAGERS = [
    "Sam Rivera", "Priya Nair", "Chris Delgado", "Nina Okonkwo", "Lee Huang",
    "Jordan Blake", "Maya Santos", "Owen Keller",
]

LEAVE_REASONS = {
    "VACATION": [
        "Personal travel",
        "Family visit",
        "Long weekend trip",
        "Holiday break",
        "Staycation",
    ],
    "SICK": [
        "Illness recovery",
        "Medical appointment",
        "Flu symptoms",
        "Outpatient procedure",
    ],
    "PERSONAL": [
        "Personal matter",
        "Home appointment",
        "Family commitment",
        "Errands / logistics",
    ],
}

PENDING_ACTION_TEMPLATES: list[tuple[str, str, str]] = [
    ("ACT-TS", "Submit timesheet", "HIGH"),
    ("ACT-BE", "Complete benefits enrollment", "HIGH"),
    ("ACT-SP", "Acknowledge security policy", "MEDIUM"),
    ("ACT-EC", "Update emergency contact", "MEDIUM"),
    ("ACT-MA", "Manager approval required", "HIGH"),
    ("ACT-EX", "Submit expense report", "MEDIUM"),
    ("ACT-TR", "Complete compliance training", "LOW"),
    ("ACT-ID", "Renew badge photo", "LOW"),
]


def employee_id_for_index(index: int) -> str:
    """Map 0-based index → ``E-1101`` … ``E-1130``."""
    return f"E-{FIRST_EMPLOYEE_NUM + index}"


def _stable_int(*parts: Any, seed: int = DEFAULT_SEED) -> int:
    """Deterministic non-negative int from seed + parts."""
    material = f"{seed}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _pick(seq: list[Any], *parts: Any, seed: int = DEFAULT_SEED) -> Any:
    """Pick a stable element from ``seq``."""
    return seq[_stable_int(*parts, seed=seed) % len(seq)]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def build_company_holidays(*, as_of: date = DEFAULT_AS_OF) -> list[CompanyHoliday]:
    """Shared company holiday calendar spanning demo years."""
    del as_of  # calendar is fixed; parameter kept for API symmetry
    return [
        CompanyHoliday(holiday_name="New Year's Day", date="2024-01-01", location="All"),
        CompanyHoliday(holiday_name="Memorial Day", date="2024-05-27", location="US"),
        CompanyHoliday(holiday_name="Independence Day", date="2024-07-04", location="US"),
        CompanyHoliday(holiday_name="Labor Day", date="2024-09-02", location="US"),
        CompanyHoliday(holiday_name="Thanksgiving", date="2024-11-28", location="US"),
        CompanyHoliday(holiday_name="Christmas Day", date="2024-12-25", location="All"),
        CompanyHoliday(holiday_name="New Year's Day", date="2025-01-01", location="All"),
        CompanyHoliday(holiday_name="Memorial Day", date="2025-05-26", location="US"),
        CompanyHoliday(holiday_name="Independence Day", date="2025-07-04", location="US"),
        CompanyHoliday(holiday_name="Labor Day", date="2025-09-01", location="US"),
        CompanyHoliday(holiday_name="Thanksgiving", date="2025-11-27", location="US"),
        CompanyHoliday(holiday_name="Christmas Day", date="2025-12-25", location="All"),
        CompanyHoliday(holiday_name="New Year's Day", date="2026-01-01", location="All"),
        CompanyHoliday(holiday_name="Memorial Day", date="2026-05-25", location="US"),
        CompanyHoliday(holiday_name="Independence Day", date="2026-07-03", location="US"),
        CompanyHoliday(holiday_name="Labor Day", date="2026-09-07", location="US"),
        CompanyHoliday(holiday_name="Thanksgiving", date="2026-11-26", location="US"),
        CompanyHoliday(holiday_name="Christmas Day", date="2026-12-25", location="All"),
        # Near-term demos around Aug 2026
        CompanyHoliday(
            holiday_name="Company Foundation Day",
            date="2026-08-14",
            location="All",
        ),
        CompanyHoliday(
            holiday_name="Summer Wellness Day",
            date="2026-08-21",
            location="All",
        ),
    ]


def _leave_history_for(
    employee_id: str,
    *,
    rich: bool,
    seed: int,
) -> list[LeaveHistoryRecord]:
    """Generate 2024–2026 leave history."""
    records: list[LeaveHistoryRecord] = []
    years = (2024, 2025, 2026)
    type_cycle = ("VACATION", "SICK", "PERSONAL", "VACATION", "SICK")

    events_per_year = 6 if rich else 3
    for year in years:
        for i in range(events_per_year):
            leave_type = type_cycle[(i + _stable_int(employee_id, year, seed=seed)) % 5]
            month = _clamp(
                1 + (_stable_int(employee_id, year, "m", i, seed=seed) % 12),
                1,
                12 if year < 2026 else 7,  # history only through Jul 2026
            )
            if year == 2026 and month > 7:
                month = 1 + (i % 7)
            day = 1 + (
                _stable_int(employee_id, year, "d", i, seed=seed)
                % monthrange(year, month)[1]
            )
            # Cap day so multi-day leave stays in-month for simplicity.
            max_days = 3 if leave_type == "VACATION" else 2
            days = 1 + (_stable_int(employee_id, year, "len", i, seed=seed) % max_days)
            if leave_type == "SICK" and days > 2:
                days = 1
            if day + days - 1 > monthrange(year, month)[1]:
                day = max(1, monthrange(year, month)[1] - days)
            status = "APPROVED"
            if rich and i == events_per_year - 1 and year == 2025:
                status = "REJECTED"
            reason = _pick(
                LEAVE_REASONS[leave_type],
                employee_id,
                year,
                i,
                "reason",
                seed=seed,
            )
            records.append(
                LeaveHistoryRecord(
                    date=date(year, month, day).isoformat(),
                    type=leave_type,  # type: ignore[arg-type]
                    days=float(days),
                    status=status,  # type: ignore[arg-type]
                    reason=reason,
                )
            )

    # Primary demo employee: explicit, narratable 2025 vacation total.
    if employee_id == "E-1101":
        records.extend(
            [
                LeaveHistoryRecord(
                    date="2025-03-10",
                    type="VACATION",
                    days=3.0,
                    status="APPROVED",
                    reason="Personal travel",
                ),
                LeaveHistoryRecord(
                    date="2025-07-21",
                    type="VACATION",
                    days=5.0,
                    status="APPROVED",
                    reason="Summer vacation",
                ),
                LeaveHistoryRecord(
                    date="2025-11-24",
                    type="VACATION",
                    days=2.0,
                    status="APPROVED",
                    reason="Thanksgiving travel",
                ),
                LeaveHistoryRecord(
                    date="2025-02-12",
                    type="SICK",
                    days=2.0,
                    status="APPROVED",
                    reason="Illness recovery",
                ),
                LeaveHistoryRecord(
                    date="2024-06-03",
                    type="VACATION",
                    days=4.0,
                    status="APPROVED",
                    reason="Family visit",
                ),
                LeaveHistoryRecord(
                    date="2026-04-15",
                    type="PERSONAL",
                    days=1.0,
                    status="APPROVED",
                    reason="Personal matter",
                ),
            ]
        )
    return sorted(records, key=lambda r: r.date)


def _upcoming_leave_for(
    employee_id: str,
    *,
    rich: bool,
    as_of: date,
    seed: int,
) -> list[UpcomingLeave]:
    """Upcoming approved / pending leave for selected employees."""
    index = int(employee_id.split("-")[1]) - FIRST_EMPLOYEE_NUM
    if not rich and index % 4 != 0:
        return []

    items: list[UpcomingLeave] = []
    if employee_id == "E-1101":
        items.append(
            UpcomingLeave(
                start_date=(as_of + timedelta(days=18)).isoformat(),
                end_date=(as_of + timedelta(days=22)).isoformat(),
                leave_type="VACATION",
                status="APPROVED",
                number_of_days=5.0,
            )
        )
        items.append(
            UpcomingLeave(
                start_date=(as_of + timedelta(days=45)).isoformat(),
                end_date=(as_of + timedelta(days=45)).isoformat(),
                leave_type="PERSONAL",
                status="PENDING",
                number_of_days=1.0,
            )
        )
        return items

    start = as_of + timedelta(
        days=10 + (_stable_int(employee_id, "up", seed=seed) % 40)
    )
    days = 1 + (_stable_int(employee_id, "updays", seed=seed) % 3)
    end = start + timedelta(days=days - 1)
    leave_type = _pick(["VACATION", "SICK", "PERSONAL"], employee_id, "ut", seed=seed)
    status = "PENDING" if index % 5 == 0 else "APPROVED"
    items.append(
        UpcomingLeave(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            leave_type=leave_type,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            number_of_days=float(days),
        )
    )
    return items


def _pending_actions_for(
    employee_id: str,
    *,
    rich: bool,
    as_of: date,
    seed: int,
) -> list[PendingAction]:
    """HR pending actions — rich set for demo employees."""
    count = 4 if rich else 1 + (_stable_int(employee_id, "pac", seed=seed) % 3)
    actions: list[PendingAction] = []
    for i in range(count):
        prefix, description, priority = PENDING_ACTION_TEMPLATES[
            (i + _stable_int(employee_id, seed=seed)) % len(PENDING_ACTION_TEMPLATES)
        ]
        due = as_of + timedelta(
            days=_stable_int(employee_id, "due", i, seed=seed) % 21 - 3
        )
        status = "OVERDUE" if due < as_of else "PENDING"
        if rich and i == 1:
            status = "IN_PROGRESS"
        actions.append(
            PendingAction(
                action_id=f"{prefix}-{employee_id[2:]}-{i + 1:02d}",
                description=description,
                due_date=due.isoformat(),
                status=status,  # type: ignore[arg-type]
                priority=priority,  # type: ignore[arg-type]
            )
        )

    if employee_id == "E-1101":
        # Guarantee narratable demo actions.
        actions = [
            PendingAction(
                action_id="ACT-TS-1101-01",
                description="Submit timesheet",
                due_date=(as_of + timedelta(days=2)).isoformat(),
                status="PENDING",
                priority="HIGH",
            ),
            PendingAction(
                action_id="ACT-SP-1101-02",
                description="Acknowledge security policy",
                due_date=(as_of + timedelta(days=5)).isoformat(),
                status="PENDING",
                priority="MEDIUM",
            ),
            PendingAction(
                action_id="ACT-EC-1101-03",
                description="Update emergency contact",
                due_date=(as_of - timedelta(days=1)).isoformat(),
                status="OVERDUE",
                priority="HIGH",
            ),
            PendingAction(
                action_id="ACT-EX-1101-04",
                description="Submit expense report",
                due_date=(as_of + timedelta(days=10)).isoformat(),
                status="IN_PROGRESS",
                priority="MEDIUM",
            ),
        ]
    return actions


def _attendance_for(
    employee_id: str,
    *,
    seed: int,
) -> list[AttendanceMonth]:
    """Monthly attendance for 2025 and 2026 (Jan–Jul for current year)."""
    rows: list[AttendanceMonth] = []
    for year in (2025, 2026):
        last_month = 12 if year == 2025 else 7
        for month in range(1, last_month + 1):
            working = 20 + (_stable_int(employee_id, year, month, "wd", seed=seed) % 3)
            leave_days = float(
                _stable_int(employee_id, year, month, "ld", seed=seed) % 3
            )
            absent = _stable_int(employee_id, year, month, "ab", seed=seed) % 2
            late = _stable_int(employee_id, year, month, "late", seed=seed) % 3
            present = max(0, working - int(leave_days) - absent)
            if employee_id == "E-1101" and year == 2026 and month == 7:
                # Narratable "last month" relative to Aug 2026 as_of.
                working, present, absent, leave_days, late = 22, 20, 0, 2.0, 1
            rows.append(
                AttendanceMonth(
                    year=year,
                    month=month,
                    working_days=working,
                    days_present=present,
                    days_absent=absent,
                    leave_days=leave_days,
                    late_days=late,
                )
            )
    return rows


def _payroll_for(employee_id: str, *, as_of: date, seed: int) -> PayrollSummary:
    """Fictional payroll — never returned without verification."""
    index = int(employee_id.split("-")[1]) - FIRST_EMPLOYEE_NUM
    base = 5200 + (index * 175) + (_stable_int(employee_id, "pay", seed=seed) % 400)
    if employee_id == "E-1101":
        base = 7250
    gross = float(base)
    net = round(gross * 0.78, 2)
    # Semi-monthly: 15th and last day approximation → next from as_of.
    if as_of.day < 15:
        last_pay = date(as_of.year, as_of.month, 1) - timedelta(days=1)
        next_pay = date(as_of.year, as_of.month, 15)
    else:
        last_pay = date(as_of.year, as_of.month, 15)
        # Next = last calendar day of month
        next_pay = date(
            as_of.year, as_of.month, monthrange(as_of.year, as_of.month)[1]
        )
    pending = None
    status = "CURRENT"
    if index % 7 == 3:
        status = "PROCESSING"
        pending = "Direct deposit update in review"
    if employee_id == "E-1101":
        status = "CURRENT"
        pending = None
    return PayrollSummary(
        monthly_gross=gross,
        monthly_net=net,
        currency="USD",
        last_pay_date=last_pay.isoformat(),
        next_pay_date=next_pay.isoformat(),
        payroll_status=status,  # type: ignore[arg-type]
        pending_payroll_action=pending,
    )


def _leave_balance_for(
    employee_id: str,
    *,
    rich: bool,
    seed: int,
) -> LeaveBalance:
    vacation = 8 + (_stable_int(employee_id, "vac", seed=seed) % 10)
    sick = 4 + (_stable_int(employee_id, "sick", seed=seed) % 7)
    personal = 1 + (_stable_int(employee_id, "pers", seed=seed) % 4)
    if employee_id == "E-1101":
        return LeaveBalance(vacation=14, sick=8, personal=3)
    if employee_id == "E-1102":
        return LeaveBalance(vacation=6, sick=10, personal=2)
    if employee_id == "E-1103":
        return LeaveBalance(vacation=18, sick=5, personal=4)
    if not rich:
        return LeaveBalance(vacation=vacation, sick=sick, personal=personal)
    return LeaveBalance(vacation=vacation, sick=sick, personal=personal)


def build_employee(
    index: int,
    *,
    seed: int = DEFAULT_SEED,
    as_of: date = DEFAULT_AS_OF,
) -> EmployeeRecord:
    """Build one deterministic employee record."""
    employee_id = employee_id_for_index(index)
    rich = index < 5  # E-1101 … E-1105
    dept, base_title = DEPARTMENTS[index % len(DEPARTMENTS)]
    designation = (
        SENIOR_TITLES[dept]
        if rich or _stable_int(employee_id, "title", seed=seed) % 3 == 0
        else base_title
    )
    if employee_id == "E-1101":
        full_name = "Avery Nguyen"
        department = "Engineering"
        designation = "Senior Software Engineer"
        manager = "Sam Rivera"
        location = "Austin"
        joining = date(2021, 3, 15)
        work_mode = "HYBRID"
    else:
        full_name = (
            f"{_pick(FIRST_NAMES, employee_id, 'fn', seed=seed)} "
            f"{_pick(LAST_NAMES, employee_id, 'ln', seed=seed)}"
        )
        department = dept
        manager = _pick(MANAGERS, employee_id, "mgr", seed=seed)
        location = _pick(LOCATIONS, employee_id, "loc", seed=seed)
        year = 2018 + (_stable_int(employee_id, "jy", seed=seed) % 7)
        month = 1 + (_stable_int(employee_id, "jm", seed=seed) % 12)
        day = 1 + (_stable_int(employee_id, "jd", seed=seed) % 28)
        joining = date(year, month, day)
        work_mode = _pick(
            ["ONSITE", "HYBRID", "REMOTE"],
            employee_id,
            "wm",
            seed=seed,
        )

    employment_status = "ACTIVE"
    if index == 27:
        employment_status = "ON_LEAVE"
    employment_type = "FULL_TIME"
    if index % 11 == 0:
        employment_type = "CONTRACT"
    elif index % 13 == 0:
        employment_type = "PART_TIME"

    return EmployeeRecord(
        employee_id=employee_id,
        full_name=full_name,
        department=department,
        designation=designation,
        manager=manager,
        location=location,
        joining_date=joining.isoformat(),
        employment_status=employment_status,  # type: ignore[arg-type]
        employment_type=employment_type,  # type: ignore[arg-type]
        work_mode=work_mode,  # type: ignore[arg-type]
        leave_balance=_leave_balance_for(employee_id, rich=rich, seed=seed),
        leave_history=_leave_history_for(employee_id, rich=rich, seed=seed),
        upcoming_leave=_upcoming_leave_for(
            employee_id, rich=rich, as_of=as_of, seed=seed
        ),
        payroll=_payroll_for(employee_id, as_of=as_of, seed=seed),
        pending_actions=_pending_actions_for(
            employee_id, rich=rich, as_of=as_of, seed=seed
        ),
        attendance=_attendance_for(employee_id, seed=seed),
    )


def generate_employee_dataset(
    *,
    employee_count: int = DEFAULT_EMPLOYEE_COUNT,
    seed: int = DEFAULT_SEED,
    as_of: date | None = None,
) -> EmployeeDataset:
    """Generate the full deterministic dataset (idempotent content)."""
    if employee_count < 10 or employee_count > 50:
        raise ValueError("employee_count must be in 10–50 for demo realism")
    as_of_date = as_of or DEFAULT_AS_OF
    employees = [
        build_employee(i, seed=seed, as_of=as_of_date)
        for i in range(employee_count)
    ]
    return EmployeeDataset(
        version=1,
        seed=seed,
        as_of_date=as_of_date.isoformat(),
        employee_count=len(employees),
        employees=employees,
        holidays=build_company_holidays(as_of=as_of_date),
    )


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_AS_OF",
    "DEFAULT_EMPLOYEE_COUNT",
    "employee_id_for_index",
    "generate_employee_dataset",
    "build_employee",
    "build_company_holidays",
]
