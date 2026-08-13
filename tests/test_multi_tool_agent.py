"""Multi-tool agent contracts, workflows, confirmation, and metadata tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.planner import RuleBasedPlanner
from app.agent.service import AgentService
from app.agent.tools.employee_tools import (
    CheckLeaveEligibilityTool,
    CreateLeaveRequestTool,
    GetHolidayCalendarTool,
    GetManagerInformationTool,
    PROTECTED_EMPLOYEE_TOOLS,
    WRITE_EMPLOYEE_TOOLS,
    build_employee_tools,
)
from app.employees.exceptions import EmployeeValidationError
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.evaluation.extension_points import (
    CI_EVALUATION_GATE_PLACEHOLDERS,
    EvaluationRoadmapCatalog,
    MetricGroup,
)
from app.evaluation.tool_validation.tool_execution import ToolExecutionStatus
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService


def _rag_response(question: str, answer: str) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_documents=[
            RetrievedDocument(
                id="chunk-1",
                text="Carry-forward vacation days are limited by handbook policy.",
                metadata={"source": "employee_handbook.pdf"},
                score=0.91,
            )
        ],
        retrieval_metadata={"retrieved_count": 1},
        generation_metadata={
            "model": "mock-llm",
            "prompt": {"system": "sys", "user": question},
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        latency=LatencyBreakdown(
            retrieval_ms=1.0,
            prompt_build_ms=1.0,
            llm_ms=2.0,
            total_ms=4.0,
        ),
        correlation_id="rag-corr-mt",
    )


@pytest.fixture
def mock_rag_service() -> MagicMock:
    service = MagicMock()
    service.query.side_effect = lambda request: _rag_response(
        request.question,
        "Carry-forward policy excerpt from the employee handbook.",
    )
    return service


@pytest.fixture
def mock_tracing() -> TracingService:
    return TracingService(tracer=NoOpTracer())


@pytest.fixture
def employee_service(tmp_path: Path) -> EmployeeService:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    return EmployeeService(store)


@pytest.fixture
def agent(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
    employee_service: EmployeeService,
) -> AgentService:
    return AgentService(
        rag_service=mock_rag_service,
        employee_service=employee_service,
        tracing_service=mock_tracing,
    )


# ---------------------------------------------------------------------------
# Tool contracts / validation
# ---------------------------------------------------------------------------


def test_build_employee_tools_includes_multi_tool_set(
    employee_service: EmployeeService,
) -> None:
    names = {t.name for t in build_employee_tools(employee_service)}
    assert {
        "get_employee_profile",
        "get_manager_information",
        "get_leave_balance",
        "get_leave_history",
        "get_holiday_calendar",
        "check_leave_eligibility",
        "create_leave_request",
    } <= names
    assert "create_leave_request" in WRITE_EMPLOYEE_TOOLS
    assert "check_leave_eligibility" in PROTECTED_EMPLOYEE_TOOLS


def test_manager_tool_contract(employee_service: EmployeeService) -> None:
    tool = GetManagerInformationTool(employee_service)
    result = tool.execute({}, context={"verified_employee_id": "E-1101"})
    assert result.success
    assert result.output["employee_id"] == "E-1101"
    assert result.output["manager"]
    assert "employee_name" in result.output


def test_holiday_calendar_tool_filters_by_year(
    employee_service: EmployeeService,
) -> None:
    tool = GetHolidayCalendarTool(employee_service)
    result = tool.execute({"country": "US", "year": 2026})
    assert result.success
    assert result.output["year"] == 2026
    assert result.output["count"] >= 1
    assert all(
        str(h["date"]).startswith("2026-") for h in result.output["holidays"]
    )


def test_check_leave_eligibility_insufficient_balance(
    employee_service: EmployeeService,
) -> None:
    tool = CheckLeaveEligibilityTool(employee_service)
    result = tool.execute(
        {"leave_type": "VACATION", "requested_days": 100},
        context={"verified_employee_id": "E-1101"},
    )
    assert result.success
    assert result.output["eligible"] is False
    assert any("Insufficient" in r for r in result.output["reasons"])


def test_check_leave_eligibility_invalid_type(
    employee_service: EmployeeService,
) -> None:
    tool = CheckLeaveEligibilityTool(employee_service)
    result = tool.execute(
        {"leave_type": "sabbatical", "requested_days": 2},
        context={"verified_employee_id": "E-1101"},
    )
    assert not result.success
    assert "Invalid leave type" in (result.error or "")


def test_create_leave_request_requires_confirmation(
    employee_service: EmployeeService,
) -> None:
    tool = CreateLeaveRequestTool(employee_service)
    denied = tool.execute(
        {
            "leave_type": "VACATION",
            "start_date": "2026-09-01",
            "end_date": "2026-09-05",
            "reason": "Trip",
            "confirmed": False,
        },
        context={"verified_employee_id": "E-1101"},
    )
    assert not denied.success
    assert "confirmation" in (denied.error or "").lower()


def test_create_leave_request_writes_when_confirmed(
    employee_service: EmployeeService,
) -> None:
    tool = CreateLeaveRequestTool(employee_service)
    created = tool.execute(
        {
            "leave_type": "VACATION",
            "start_date": "2026-09-01",
            "end_date": "2026-09-03",
            "reason": "Family trip",
            "confirmed": True,
        },
        context={"verified_employee_id": "E-1101"},
    )
    assert created.success
    assert created.output["created"] is True
    assert created.output["status"] == "PENDING"
    upcoming = employee_service.get_upcoming_leave("E-1101")
    assert any(
        item.get("start_date") == "2026-09-01" for item in upcoming["upcoming_leave"]
    )


def test_service_rejects_invalid_dates(employee_service: EmployeeService) -> None:
    with pytest.raises(EmployeeValidationError):
        employee_service.create_leave_request(
            "E-1101",
            leave_type="VACATION",
            start_date="2026-09-10",
            end_date="2026-09-01",
            reason="Bad window",
            confirmed=True,
        )


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def test_leave_eligibility_workflow(agent: AgentService) -> None:
    result = agent.run(
        "Can I take 10 days of vacation?",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert tools[:3] == [
        "get_employee_profile",
        "get_leave_balance",
        "check_leave_eligibility",
    ]
    assert "search_company_policy" in tools  # 10 days → policy context
    assert all(e.success for e in result.tool_executions)
    assert "eligib" in result.final_answer.lower()
    assert "create_leave_request" not in tools


def test_carry_forward_workflow(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run(
        "Can I carry forward my remaining vacation?",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert tools == ["get_leave_balance", "search_company_policy"]
    assert mock_rag_service.query.called
    assert "vacation" in result.final_answer.lower()
    assert result.metadata.get("rag_used") is True


def test_leave_request_requires_human_confirmation(agent: AgentService) -> None:
    draft = agent.run(
        "I want to take 5 days of vacation next month.",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in draft.tool_executions]
    assert "check_leave_eligibility" in tools
    assert "create_leave_request" not in tools
    assert draft.metadata.get("awaiting_confirmation") is True
    pending = draft.metadata.get("pending_leave_request")
    assert isinstance(pending, dict)
    assert pending.get("leave_type") == "VACATION"
    assert "confirm" in draft.final_answer.lower()

    # Without confirmation metadata, create must not run.
    bare_yes = agent.run(
        "confirm",
        metadata={"employee_id": "E-1101"},
    )
    assert "create_leave_request" not in [
        e.tool_name for e in bare_yes.tool_executions
    ]

    confirmed = agent.run(
        "confirm",
        metadata={
            "employee_id": "E-1101",
            "pending_leave_request": pending,
        },
    )
    conf_tools = [e.tool_name for e in confirmed.tool_executions]
    assert conf_tools == ["create_leave_request"]
    assert confirmed.tool_executions[0].success
    assert confirmed.metadata.get("awaiting_confirmation") is False


def test_leave_request_cancel_clears_pending(agent: AgentService) -> None:
    draft = agent.run(
        "I want to take 3 days of vacation next month.",
        metadata={"employee_id": "E-1101"},
    )
    pending = draft.metadata.get("pending_leave_request")
    cancelled = agent.run(
        "cancel",
        metadata={
            "employee_id": "E-1101",
            "pending_leave_request": pending,
        },
    )
    assert cancelled.tool_executions == []
    assert "cancel" in cancelled.final_answer.lower()


def test_manager_and_holiday_workflow(agent: AgentService) -> None:
    result = agent.run(
        "Who is my manager and what holidays are coming up?",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert tools == ["get_manager_information", "get_holiday_calendar"]
    assert all(e.success for e in result.tool_executions)
    lowered = result.final_answer.lower()
    assert "manager" in lowered
    assert "holiday" in lowered


def test_verification_still_gates_multi_tool_workflows(agent: AgentService) -> None:
    result = agent.run("Can I take 5 days of vacation?")
    assert result.tool_executions == []
    assert "verify" in result.final_answer.lower()


def test_partial_multi_tool_failure_surfaces_friendly_copy(
    agent: AgentService,
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.side_effect = RuntimeError("vector store unavailable")
    result = agent.run(
        "Can I carry forward my remaining vacation?",
        metadata={"employee_id": "E-1101"},
    )
    tools = {e.tool_name: e for e in result.tool_executions}
    assert tools["get_leave_balance"].success
    assert not tools["search_company_policy"].success
    assert "OPENAI_API_KEY" not in result.final_answer
    assert "vector store" not in result.final_answer.lower()
    assert "vacation" in result.final_answer.lower() or "trouble" in result.final_answer.lower()


def test_developer_mode_metadata_for_multi_tool(agent: AgentService) -> None:
    result = agent.run(
        "Who is my manager and what holidays are coming up?",
        metadata={"employee_id": "E-1101", "api_key": "sk-secret"},
    )
    meta = result.metadata
    assert meta.get("original_input")
    assert meta.get("normalized_input")
    assert meta.get("detected_intent")
    assert meta.get("selected_route") == "MULTI_TOOL"
    assert meta.get("verification_status") == "verified"
    assert meta.get("tool_execution_order") == [
        "get_manager_information",
        "get_holiday_calendar",
    ]
    assert isinstance(meta.get("tools_invoked"), list)
    assert len(meta["tools_invoked"]) == 2
    assert meta["tools_invoked"][0]["arguments"] is not None
    assert "api_key" not in meta
    assert meta.get("latency_ms") is not None


def test_planner_does_not_auto_write_without_confirm() -> None:
    planned = RuleBasedPlanner().plan(
        "I want to take 5 days of vacation next month.",
        metadata={"employee_id": "E-1101"},
    )
    assert "create_leave_request" not in planned.execution_order
    assert planned.pending_action is not None
    assert planned.pending_action.get("awaiting_confirmation") is True


def test_evaluation_extension_points_catalog() -> None:
    catalog = EvaluationRoadmapCatalog()
    groups = set(catalog.list_groups())
    assert MetricGroup.AGENT_TOOL_CALLING.value in groups
    assert MetricGroup.SAFETY.value in groups
    assert MetricGroup.NL2SQL.value in groups
    names = catalog.metric_names()
    assert "faithfulness" in names
    assert "human_confirmation_compliance" in names
    assert "tool_selection_accuracy" in names
    assert CI_EVALUATION_GATE_PLACEHOLDERS["faithfulness"] == 0.70
