"""Tests for deterministic employee dataset, store, and verification boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent.planner import RuleBasedPlanner
from app.agent.tools.employee_tools import (
    GetLeaveBalanceTool,
    GetPayrollSummaryTool,
    build_employee_tools,
)
from app.employees.exceptions import EmployeeNotVerifiedError, EmployeeVerificationError
from app.employees.generator import (
    DEFAULT_EMPLOYEE_COUNT,
    DEFAULT_SEED,
    generate_employee_dataset,
)
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.main import create_app


def test_generate_dataset_is_deterministic() -> None:
    a = generate_employee_dataset(employee_count=30, seed=DEFAULT_SEED)
    b = generate_employee_dataset(employee_count=30, seed=DEFAULT_SEED)
    assert a.model_dump() == b.model_dump()
    assert a.employee_count == DEFAULT_EMPLOYEE_COUNT
    assert [e.employee_id for e in a.employees] == [
        f"E-{n}" for n in range(1101, 1131)
    ]


def test_primary_demo_employee_has_rich_data() -> None:
    dataset = generate_employee_dataset()
    demo = dataset.by_id()["E-1101"]
    assert demo.full_name == "Avery Nguyen"
    assert demo.department == "Engineering"
    assert demo.leave_balance.vacation == 14
    assert demo.leave_balance.sick == 8
    assert demo.leave_balance.personal == 3
    assert any(h.date.startswith("2025-") for h in demo.leave_history)
    assert any(h.date.startswith("2024-") for h in demo.leave_history)
    assert any(h.date.startswith("2026-") for h in demo.leave_history)
    assert demo.upcoming_leave
    assert demo.pending_actions
    assert demo.payroll.monthly_gross > 0
    assert len(demo.attendance) >= 12
    assert dataset.holidays


def test_seed_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "employees.json"
    store = EmployeeStore(path)
    first = store.seed()
    mtime_1 = path.stat().st_mtime_ns
    second = store.seed()
    mtime_2 = path.stat().st_mtime_ns
    assert first.employee_count == second.employee_count == 30
    assert mtime_1 == mtime_2
    force = store.seed(force=True)
    assert force.employee_count == 30
    assert path.stat().st_mtime_ns >= mtime_2


def test_verification_boundary_blocks_unverified_access(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    service = EmployeeService(store)

    with pytest.raises(EmployeeNotVerifiedError):
        service.get_leave_balance(None)
    with pytest.raises(EmployeeNotVerifiedError):
        service.get_payroll_summary(None)
    with pytest.raises(EmployeeNotVerifiedError):
        service.get_employee_profile("")

    verified = service.verify_employee("E-1101")
    assert verified.verified is True
    balance = service.get_leave_balance("E-1101")
    assert balance["leave_balance"]["vacation"] == 14
    payroll = service.get_payroll_summary("E-1101")
    assert payroll["payroll"]["currency"] == "USD"


def test_verify_rejects_unknown_employee(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    service = EmployeeService(store)
    with pytest.raises(EmployeeVerificationError):
        service.verify_employee("E-9999")


def test_protected_tools_ignore_spoofed_employee_id(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    service = EmployeeService(store)
    tool = GetLeaveBalanceTool(service)

    # Session is E-1101; LLM tries to request E-1102 — must stay on E-1101.
    result = tool.execute(
        {"employee_id": "E-1102"},
        context={"verified_employee_id": "E-1101"},
    )
    assert result.success
    assert result.output["employee_id"] == "E-1101"
    assert result.output["leave_balance"]["vacation"] == 14


def test_payroll_tool_requires_verification(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    tool = GetPayrollSummaryTool(EmployeeService(store))
    failed = tool.execute({}, context={})
    assert not failed.success
    assert "verify" in (failed.error or "").lower()


def test_holidays_do_not_require_verification(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    service = EmployeeService(store)
    holidays = service.get_upcoming_holidays()
    assert holidays["next_holiday"] is not None
    assert holidays["upcoming_holidays"]


def test_planner_routes_employee_and_hybrid_intents() -> None:
    planner = RuleBasedPlanner()
    leave = planner.plan(
        "How many vacation days do I have left?",
        metadata={"employee_id": "E-1101"},
    )
    assert leave.execution_order == ["get_leave_balance"]

    hybrid = planner.plan(
        "Can I carry forward my remaining vacation days?",
        metadata={"employee_id": "E-1101"},
    )
    assert "get_leave_balance" in hybrid.execution_order
    assert "search_company_policy" in hybrid.execution_order
    assert "search_docs" not in hybrid.execution_order

    holidays = planner.plan("What is the next company holiday?")
    assert holidays.execution_order == ["get_upcoming_holidays"]


def test_verify_api_endpoint(tmp_path: Path) -> None:
    path = tmp_path / "employees.json"
    store = EmployeeStore(path)
    store.seed()
    service = EmployeeService(store)

    from app.api.deps import get_employee_service

    app = create_app()
    app.dependency_overrides[get_employee_service] = lambda: service
    client = TestClient(app)

    ok = client.post("/api/v1/employees/verify", json={"employee_id": "E-1101"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["verified"] is True
    assert body["employee_id"] == "E-1101"
    assert body["full_name"] == "Avery Nguyen"

    bad = client.post("/api/v1/employees/verify", json={"employee_id": "E-9999"})
    assert bad.status_code == 400
    app.dependency_overrides.clear()


def test_build_employee_tools_registers_expected_names(tmp_path: Path) -> None:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    tools = build_employee_tools(EmployeeService(store))
    names = {t.name for t in tools}
    assert {
        "verify_employee",
        "get_employee_profile",
        "get_leave_balance",
        "get_leave_history",
        "get_manager_information",
        "get_holiday_calendar",
        "check_leave_eligibility",
        "create_leave_request",
        "get_upcoming_leave",
        "get_pending_actions",
        "get_payroll_summary",
        "get_attendance_summary",
        "get_upcoming_holidays",
    } <= names
