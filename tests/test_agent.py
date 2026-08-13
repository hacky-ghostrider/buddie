"""Sprint 11 — LangGraph agent tests (no live API calls)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.graph import build_agent_graph
from app.agent.planner import Planner, RuleBasedPlanner
from app.agent.router import ToolRouter
from app.agent.service import AgentService
from app.agent.state import AgentState, empty_agent_state
from app.agent.tools import (
    CalculatorTool,
    SearchTool,
    build_default_tool_registry,
)
from app.agent.tools.calculator_tool import evaluate_arithmetic
from app.agent.tools.rag_tool import RAGToolBundle
from app.evaluation.context import EvaluationContext
from app.evaluation.scenarios import (
    CANONICAL_SCENARIO_ID,
    SPRINT_11_EXPECTED_ACTUAL_TOOLS,
)
from app.evaluation.tool_validation.tool_contract import ToolContract
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionStatus,
)
from app.evaluation.tool_validation.trace_mapper import ToolTraceMapper
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
                text="Employees accrue paid leave per the handbook policy.",
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
        correlation_id="rag-corr-1",
    )


@pytest.fixture
def mock_rag_service() -> MagicMock:
    service = MagicMock()
    service.query.side_effect = lambda request: _rag_response(
        request.question,
        "The employee handbook leave policy defines paid time off eligibility, "
        "accrual, approval workflow, and notice requirements.",
    )
    return service


@pytest.fixture
def mock_tracing() -> TracingService:
    return TracingService(tracer=NoOpTracer())


# ---------------------------------------------------------------------------
# ToolExecutionStatus
# ---------------------------------------------------------------------------


def test_tool_execution_status_enum_values() -> None:
    assert ToolExecutionStatus.SUCCESS.value == "success"
    assert ToolExecutionStatus.FAILED.value == "failed"
    assert ToolExecutionStatus.SKIPPED.value == "skipped"
    assert ToolExecutionStatus.TIMEOUT.value == "timeout"
    assert ToolExecutionStatus.RETRY.value == "retry"
    assert ToolExecutionStatus.CANCELLED.value == "cancelled"


def test_tool_execution_status_coerce_aliases() -> None:
    assert ToolExecutionStatus.coerce("error") == ToolExecutionStatus.FAILED
    assert ToolExecutionStatus.coerce("ok") == ToolExecutionStatus.SUCCESS
    assert ToolExecutionStatus.coerce(ToolExecutionStatus.TIMEOUT) == (
        ToolExecutionStatus.TIMEOUT
    )


# ---------------------------------------------------------------------------
# Calculator / Search tools
# ---------------------------------------------------------------------------


def test_calculator_tool_arithmetic() -> None:
    tool = CalculatorTool()
    result = tool.execute({"expression": "(2 + 3) * 4"}, order=0)
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.output["result"] == 20
    assert evaluate_arithmetic("10 / 2") == 5.0


def test_calculator_tool_rejects_unsafe_expression() -> None:
    tool = CalculatorTool()
    result = tool.execute({"expression": "__import__('os').system('x')"}, order=0)
    assert result.status == ToolExecutionStatus.FAILED
    assert result.error


def test_search_tool_mock() -> None:
    tool = SearchTool()
    result = tool.execute({"query": "leave policy"}, order=0)
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.output["provider"] == "mock"
    assert result.output["results"]


# ---------------------------------------------------------------------------
# RAG tools (mocked RAGService)
# ---------------------------------------------------------------------------


def test_rag_tools_reuse_rag_service(mock_rag_service: MagicMock) -> None:
    bundle = RAGToolBundle(mock_rag_service)
    ctx: dict[str, Any] = {
        "question": "Summarize the leave policy from the employee handbook."
    }
    search = bundle.search_docs.execute(
        {"query": "leave policy employee handbook"},
        order=0,
        context=ctx,
    )
    summary = bundle.summarize.execute(
        {"document": "employee_handbook.pdf"},
        order=1,
        context=ctx,
    )
    assert search.status == ToolExecutionStatus.SUCCESS
    assert summary.status == ToolExecutionStatus.SUCCESS
    assert "leave" in summary.output["summary"].lower()
    assert mock_rag_service.query.call_count == 1  # summarize reuses cache


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def test_planner_canonical_leave_policy() -> None:
    planner = RuleBasedPlanner()
    planned = planner.plan(
        "Summarize the leave policy from the employee handbook.",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )
    assert planned.execution_order == ["search_docs", "summarize"]
    assert planned.required_tools == ["search_docs", "summarize"]
    assert [c.tool_name for c in planned.tool_contracts] == [
        "search_docs",
        "summarize",
    ]
    assert planned.invocations[0].arguments["query"] == (
        "leave policy employee handbook"
    )
    assert planned.invocations[1].arguments["document"] == "employee_handbook.pdf"


def test_planner_calculator_path() -> None:
    planned = RuleBasedPlanner().plan("12 * (3 + 1)")
    assert planned.selected_tools == ["calculator"]


def test_planner_node_updates_state() -> None:
    planner = Planner()
    state = empty_agent_state(
        question="Summarize the leave policy from the employee handbook.",
        correlation_id="c-1",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )
    update = planner(state)
    assert update["selected_tools"] == ["search_docs", "summarize"]
    assert update["planner_output"]["required_tools"] == [
        "search_docs",
        "summarize",
    ]


# ---------------------------------------------------------------------------
# Router + AgentState
# ---------------------------------------------------------------------------


def test_router_executes_tools_and_populates_state(
    mock_rag_service: MagicMock,
) -> None:
    registry = build_default_tool_registry(mock_rag_service)
    router = ToolRouter(registry)
    planned = RuleBasedPlanner().plan(
        "Summarize the leave policy from the employee handbook.",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )
    state = empty_agent_state(
        question="Summarize the leave policy from the employee handbook.",
        correlation_id="c-2",
    )
    state["planner_output"] = planned.model_dump(mode="json")
    update = router(state)
    history = [ToolExecution.model_validate(item) for item in update["tool_execution_history"]]
    assert [e.tool_name for e in history] == ["search_docs", "summarize"]
    assert all(e.status == ToolExecutionStatus.SUCCESS for e in history)
    assert "leave" in update["final_answer"].lower()


def test_agent_state_seed_fields() -> None:
    state = empty_agent_state(question="hello?", correlation_id="cid")
    assert state["question"] == "hello?"
    assert state["correlation_id"] == "cid"
    assert state["messages"][0]["role"] == "user"
    assert state["tool_execution_history"] == []


# ---------------------------------------------------------------------------
# Tool contracts + ToolTraceMapper
# ---------------------------------------------------------------------------


def test_planner_generates_tool_contracts() -> None:
    planned = RuleBasedPlanner().plan(
        "Summarize the leave policy from the employee handbook."
    )
    assert isinstance(planned.tool_contracts[0], ToolContract)
    assert planned.tool_contracts[0].required == ["query"]
    assert planned.tool_contracts[0].maximum_calls == 1
    assert planned.tool_contracts[1].expected_output_type == "dict"


def test_tool_trace_mapper_from_agent_langsmith_payload(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
) -> None:
    agent = AgentService(
        rag_service=mock_rag_service,
        tracing_service=mock_tracing,
    )
    result = agent.run(
        "Summarize the leave policy from the employee handbook.",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )
    payload = AgentService._executions_to_langsmith_payload(
        result.tool_executions,
        planner_output=result.planner_output,
        correlation_id=result.correlation_id,
    )
    mapped = ToolTraceMapper().map_langsmith_trace(payload)
    # Mapper includes planner child run + tools; filter tool runs.
    tool_names = [
        e.tool_name
        for e in mapped
        if e.tool_name in {"search_docs", "summarize", "calculator", "search"}
    ]
    assert tool_names == ["search_docs", "summarize"]


# ---------------------------------------------------------------------------
# EvaluationContext integration
# ---------------------------------------------------------------------------


def test_agent_populates_evaluation_context(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
) -> None:
    agent = AgentService(
        rag_service=mock_rag_service,
        tracing_service=mock_tracing,
    )
    result = agent.run(
        "Summarize the leave policy from the employee handbook.",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
        expected_answer="leave policy summary",
    )
    assert isinstance(result.evaluation_context, EvaluationContext)
    assert result.evaluation_context.question.startswith("Summarize")
    assert [t.tool_name for t in result.evaluation_context.tool_calls] == [
        "search_docs",
        "summarize",
    ]
    assert result.evaluation_context.answer
    assert result.trace_id
    assert result.run_id


# ---------------------------------------------------------------------------
# Canonical demo agent-tools-foundation-001
# ---------------------------------------------------------------------------


def test_canonical_demo_agent_tools_foundation_001(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
) -> None:
    agent = AgentService(
        rag_service=mock_rag_service,
        tracing_service=mock_tracing,
    )
    question = "Summarize the leave policy from the employee handbook."
    result = agent.run(
        question,
        metadata={
            "scenario": CANONICAL_SCENARIO_ID,
            "golden_id": CANONICAL_SCENARIO_ID,
        },
        expected_answer=(
            "The employee handbook leave policy defines paid time off eligibility, "
            "accrual, approval workflow, and notice requirements for vacation and "
            "related leave types."
        ),
        expected_sources=["employee_handbook.pdf"],
    )

    actual_tools = [e.tool_name for e in result.tool_executions]
    assert actual_tools == SPRINT_11_EXPECTED_ACTUAL_TOOLS
    assert result.planner_output is not None
    assert result.planner_output.execution_order == ["search_docs", "summarize"]
    assert result.tool_validation is not None
    assert result.tool_validation.passed is True
    assert result.tool_validation.expected_tools == ["search_docs", "summarize"]
    assert result.tool_validation.actual_tools == ["search_docs", "summarize"]
    assert result.final_answer
    assert result.evaluation_context is not None
    assert result.evaluation_context.correlation_id == result.correlation_id


def test_langgraph_compile_and_invoke(mock_rag_service: MagicMock) -> None:
    registry = build_default_tool_registry(mock_rag_service)
    graph = build_agent_graph(planner=Planner(), router=ToolRouter(registry))
    state = empty_agent_state(
        question="Summarize the leave policy from the employee handbook.",
        correlation_id="graph-1",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )
    final: AgentState = graph.invoke(state)
    assert final.get("final_answer")
    assert final.get("evaluation_context") is not None
    history = final.get("tool_execution_history") or []
    assert [h["tool_name"] for h in history] == ["search_docs", "summarize"]


def test_agent_calculator_end_to_end(
    mock_rag_service: MagicMock,
    mock_tracing: TracingService,
) -> None:
    agent = AgentService(
        rag_service=mock_rag_service,
        tracing_service=mock_tracing,
    )
    result = agent.run("2 + 2")
    assert [e.tool_name for e in result.tool_executions] == ["calculator"]
    assert result.final_answer == "4"
