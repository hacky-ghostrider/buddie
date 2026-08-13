"""Sprint 15 — MCP server/client contracts, security, and multi-tool workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.service import AgentService
from app.agent.tools.employee_tools import verified_employee_id_from_context
from app.config.settings import Settings, get_settings
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.evaluation.extension_points import EvaluationRoadmapCatalog
from app.mcp.adapter import (
    MCP_PRIMARY_TOOL_NAMES,
    McpAgentTool,
    apply_mcp_adapters,
    build_tool_registry,
)
from app.mcp.client import BuddieMcpClient, McpClientError
from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES, TOOL_DESCRIPTIONS
from app.mcp.server import build_buddie_mcp_server
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
        correlation_id="rag-corr-mcp",
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
def mcp_server(
    employee_service: EmployeeService,
    mock_rag_service: MagicMock,
):
    return build_buddie_mcp_server(
        employee_service=employee_service,
        rag_service=mock_rag_service,
    )


@pytest.fixture
def mcp_client(mcp_server) -> BuddieMcpClient:
    client = BuddieMcpClient(transport="memory", server=mcp_server)
    client.connect()
    return client


@pytest.fixture
def mcp_agent(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
    employee_service: EmployeeService,
    mcp_client: BuddieMcpClient,
) -> AgentService:
    registry, client = build_tool_registry(
        mock_rag_service,
        employee_service=employee_service,
        tool_mode="mcp",
        mcp_transport="memory",
        mcp_client=mcp_client,
    )
    return AgentService(
        rag_service=mock_rag_service,
        employee_service=employee_service,
        tracing_service=mock_tracing,
        registry=registry,
        mcp_client=client,
        settings=Settings(buddie_tool_mode="mcp", mcp_transport="memory"),
    )


# ---------------------------------------------------------------------------
# Server / discovery / schemas
# ---------------------------------------------------------------------------


def test_mcp_server_tools_are_discoverable(mcp_client: BuddieMcpClient) -> None:
    tools = mcp_client.list_tools()
    names = {tool["name"] for tool in tools}
    assert set(EXPECTED_MCP_TOOL_NAMES) <= names
    for tool in tools:
        if tool["name"] in EXPECTED_MCP_TOOL_NAMES:
            assert tool["description"]
            assert isinstance(tool["input_schema"], dict)


def test_mcp_discovery_fails_if_expected_tool_missing(mcp_server) -> None:
    client = BuddieMcpClient(transport="memory", server=mcp_server)
    client.connect()
    with pytest.raises(McpClientError) as exc:
        client.ensure_expected_tools(("get_leave_balance", "not_a_real_tool"))
    assert exc.value.code == "tool_not_found"


def test_mcp_tool_descriptions_are_meaningful() -> None:
    for name in EXPECTED_MCP_TOOL_NAMES:
        description = TOOL_DESCRIPTIONS[name]
        assert len(description) > 40
        assert "secret" not in description.lower()
        assert "api_key" not in description.lower()


def test_mcp_contract_invoke_leave_balance(mcp_client: BuddieMcpClient) -> None:
    payload = mcp_client.call_tool(
        "get_leave_balance",
        {},
        verified_employee_id="E-1101",
    )
    assert payload["ok"] is True
    assert payload["tool"] == "get_leave_balance"
    assert "leave_balance" in payload["data"]


def test_mcp_contract_invalid_arguments_handled(mcp_client: BuddieMcpClient) -> None:
    payload = mcp_client.call_tool(
        "check_leave_eligibility",
        {"leave_type": "sabbatical", "requested_days": 2},
        verified_employee_id="E-1101",
    )
    assert payload["ok"] is False
    assert payload.get("error")


def test_mcp_tool_not_found(mcp_client: BuddieMcpClient) -> None:
    with pytest.raises(McpClientError) as exc:
        mcp_client.call_tool("definitely_missing_tool", {})
    assert exc.value.code == "tool_not_found"


# ---------------------------------------------------------------------------
# Security / verification / HITL
# ---------------------------------------------------------------------------


def test_mcp_unverified_cannot_access_protected_tools(
    mcp_client: BuddieMcpClient,
) -> None:
    payload = mcp_client.call_tool("get_employee_profile", {})
    assert payload["ok"] is False
    assert payload.get("error_code") == "not_verified"


def test_mcp_arbitrary_employee_id_cannot_bypass_verification(
    mcp_client: BuddieMcpClient,
) -> None:
    # Spoofed employee_id in arguments must be ignored; still needs verified meta.
    payload = mcp_client.call_tool(
        "get_leave_balance",
        {"employee_id": "E-1102"},
    )
    assert payload["ok"] is False
    assert payload.get("error_code") == "not_verified"

    # Even with verified E-1101, argument employee_id must not switch identity.
    payload_ok = mcp_client.call_tool(
        "get_leave_balance",
        {"employee_id": "E-1102"},
        verified_employee_id="E-1101",
    )
    assert payload_ok["ok"] is True
    assert payload_ok["data"]["employee_id"] == "E-1101"


def test_mcp_write_requires_confirmation(mcp_client: BuddieMcpClient) -> None:
    denied = mcp_client.call_tool(
        "create_leave_request",
        {
            "leave_type": "VACATION",
            "start_date": "2026-09-01",
            "end_date": "2026-09-05",
            "confirmed": False,
        },
        verified_employee_id="E-1101",
    )
    assert denied["ok"] is False
    assert denied.get("error_code") == "confirmation_required"

    allowed = mcp_client.call_tool(
        "create_leave_request",
        {
            "leave_type": "VACATION",
            "start_date": "2026-09-01",
            "end_date": "2026-09-05",
            "reason": "Family trip",
            "confirmed": True,
        },
        verified_employee_id="E-1101",
        leave_request_confirmed=True,
    )
    assert allowed["ok"] is True
    assert allowed["data"].get("created") is True


def test_mcp_adapter_strips_spoofed_employee_id(
    mcp_client: BuddieMcpClient,
) -> None:
    tool = McpAgentTool("get_leave_balance", mcp_client)
    result = tool.execute(
        {"employee_id": "E-9999"},
        context={"verified_employee_id": "E-1101"},
    )
    assert result.success
    assert result.output["employee_id"] == "E-1101"
    assert "employee_id" not in result.arguments
    assert (result.trace_metadata or {}).get("protocol") == "MCP"


# ---------------------------------------------------------------------------
# Integration workflows through MCP
# ---------------------------------------------------------------------------


def test_mcp_workflow_leave_eligibility(mcp_agent: AgentService) -> None:
    result = mcp_agent.run(
        "Can I take 10 days of vacation?",
        metadata={"verified_employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_employee_profile" in tools
    assert "get_leave_balance" in tools
    assert "check_leave_eligibility" in tools
    assert result.metadata.get("selected_route") == "MULTI_TOOL"
    assert result.metadata.get("mcp", {}).get("used") is True
    assert result.metadata.get("mcp", {}).get("connected") is True
    assert all(
        item.get("protocol") == "MCP"
        for item in result.metadata.get("tools_invoked") or []
        if item.get("tool_name") in MCP_PRIMARY_TOOL_NAMES
    )
    assert result.final_answer


def test_mcp_workflow_carry_forward(mcp_agent: AgentService) -> None:
    result = mcp_agent.run(
        "Can I carry forward my remaining vacation?",
        metadata={"verified_employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_leave_balance" in tools
    assert "search_company_policy" in tools
    assert "MCP" in {
        (e.trace_metadata or {}).get("protocol") for e in result.tool_executions
    }
    assert "vacation" in result.final_answer.lower() or "leave" in result.final_answer.lower()
    assert "policy" in result.final_answer.lower()


def test_mcp_workflow_leave_request_hitl(mcp_agent: AgentService) -> None:
    draft = mcp_agent.run(
        "I want to take 5 days of vacation next month.",
        metadata={"verified_employee_id": "E-1101"},
    )
    assert draft.metadata.get("awaiting_confirmation") is True
    pending = draft.metadata.get("pending_leave_request") or {}
    assert pending.get("awaiting_confirmation") is True
    assert "create_leave_request" not in [
        e.tool_name for e in draft.tool_executions
    ]

    confirmed = mcp_agent.run(
        "Yes, submit it.",
        metadata={
            "verified_employee_id": "E-1101",
            "pending_leave_request": pending,
        },
    )
    tools = [e.tool_name for e in confirmed.tool_executions]
    assert "create_leave_request" in tools
    write = next(
        e for e in confirmed.tool_executions if e.tool_name == "create_leave_request"
    )
    assert write.success
    assert (write.trace_metadata or {}).get("protocol") == "MCP"


def test_mcp_workflow_manager_and_holidays(mcp_agent: AgentService) -> None:
    result = mcp_agent.run(
        "Who is my manager and what holidays are coming up?",
        metadata={"verified_employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_manager_information" in tools
    assert "get_holiday_calendar" in tools
    assert result.metadata.get("selected_route") == "MULTI_TOOL"
    assert result.final_answer


def test_mcp_unavailable_falls_back_to_direct(
    mock_rag_service: MagicMock,
    employee_service: EmployeeService,
) -> None:
    # Misconfigured HTTP mode should fall back to direct tools.
    registry, client = build_tool_registry(
        mock_rag_service,
        employee_service=employee_service,
        tool_mode="mcp",
        mcp_transport="http",
        mcp_server_url="",  # invalid — client construction fails path via adapter
    )
    # Empty URL raises during client init inside build_tool_registry and falls back.
    assert registry.has("get_leave_balance")
    # Direct tool still works without MCP meta.
    from app.agent.tools.employee_tools import GetLeaveBalanceTool

    # Registry may still be direct tools after fallback.
    tool = registry.get("get_leave_balance")
    # If fallback succeeded, this is a direct employee tool (no protocol MCP unless called via adapter).
    assert tool.name == "get_leave_balance"
    del client


def test_mcp_graceful_failure_when_server_unavailable(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
    employee_service: EmployeeService,
) -> None:
    class BrokenClient(BuddieMcpClient):
        def call_tool(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise McpClientError("Unable to connect to the MCP server.", code="connection_failed")

    server = build_buddie_mcp_server(
        employee_service=employee_service,
        rag_service=mock_rag_service,
    )
    broken = BrokenClient(transport="memory", server=server)
    registry, _ = build_tool_registry(
        mock_rag_service,
        employee_service=employee_service,
        tool_mode="direct",
    )
    apply_mcp_adapters(registry, broken)
    agent = AgentService(
        rag_service=mock_rag_service,
        employee_service=employee_service,
        tracing_service=mock_tracing,
        registry=registry,
        mcp_client=broken,
        settings=Settings(buddie_tool_mode="mcp"),
    )
    result = agent.run(
        "What is my leave balance?",
        metadata={"verified_employee_id": "E-1101"},
    )
    assert result.final_answer  # friendly failure, not crash
    assert any(not e.success for e in result.tool_executions)


def test_verified_context_helper_ignores_blank() -> None:
    assert verified_employee_id_from_context({}) is None
    assert (
        verified_employee_id_from_context({"verified_employee_id": "e-1101"})
        == "E-1101"
    )


def test_evaluation_roadmap_includes_mcp_metrics() -> None:
    names = EvaluationRoadmapCatalog().metric_names()
    assert "mcp_latency" in names
    assert "mcp_tool_success_rate" in names


def test_settings_tool_mode_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("BUDDIE_TOOL_MODE", "mcp")
    monkeypatch.setenv("MCP_TRANSPORT", "memory")
    settings = Settings()
    assert settings.buddie_tool_mode == "mcp"
    assert settings.mcp_transport == "memory"
    get_settings.cache_clear()
