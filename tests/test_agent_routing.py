"""Routing, failure-handling, and verification-gate tests for Buddie."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.sanity

from app.agent.conversation import (
    VERIFY_PROMPT,
    classify_intent,
    IntentRoute,
    sanitize_user_facing_answer,
)
from app.agent.planner import RuleBasedPlanner
from app.agent.service import AgentService
from app.agent.tools.employee_tools import (
    GetLeaveBalanceTool,
    PROTECTED_EMPLOYEE_TOOLS,
    verified_employee_id_from_context,
)
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.evaluation.tool_validation.tool_execution import ToolExecutionStatus
from app.generation.models import BuiltPrompt
from app.generation.offline_provider import OfflineExtractiveProvider
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
        "Paid leave policy excerpt from the employee handbook.",
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


def _assert_no_tools(result: Any, *, rag: MagicMock | None = None) -> None:
    assert result.tool_executions == []
    assert result.planner_output is not None
    assert result.planner_output.execution_order == []
    assert result.planner_output.direct_answer
    if rag is not None:
        rag.query.assert_not_called()


# ---------------------------------------------------------------------------
# Conversational
# ---------------------------------------------------------------------------


def test_greeting_hi(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("hi")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "buddie" in result.final_answer.lower()
    assert "help" in result.final_answer.lower()
    assert result.metadata.get("intent_route") == IntentRoute.CONVERSATION.value


def test_greeting_hello(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("hello")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "hello" in result.final_answer.lower()
    assert "OPENAI_API_KEY" not in result.final_answer
    assert "employee_handbook" not in result.final_answer.lower()


def test_good_morning(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("good morning")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "good morning" in result.final_answer.lower()


def test_thanks(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("thanks buddie")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "welcome" in result.final_answer.lower()


def test_goodbye(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("goodbye")
    _assert_no_tools(result, rag=mock_rag_service)
    lowered = result.final_answer.lower()
    assert "bye" in lowered or "day" in lowered


def test_cool_bye_is_goodbye(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("cool bye")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "bye" in result.final_answer.lower()
    assert result.metadata.get("intent_route") == IntentRoute.CONVERSATION.value


def test_who_are_you(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("who are you?")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "buddie" in result.final_answer.lower()
    assert "employee" in result.final_answer.lower()


def test_what_can_you_do(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("what can you do?")
    _assert_no_tools(result, rag=mock_rag_service)
    lowered = result.final_answer.lower()
    assert "leave" in lowered
    assert "policies" in lowered or "policy" in lowered


# ---------------------------------------------------------------------------
# Employee
# ---------------------------------------------------------------------------


def test_employee_request_requires_verification(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("How many vacation days do I have?")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "verify" in result.final_answer.lower()
    assert "E-1101" in result.final_answer
    assert result.planner_output.intent_route == IntentRoute.EMPLOYEE.value


def test_missing_employee_id(agent: AgentService) -> None:
    planned = RuleBasedPlanner().plan("Show my leave history.")
    assert planned.direct_answer == VERIFY_PROMPT
    assert planned.execution_order == []
    assert "get_leave_history" not in planned.required_tools
    assert "verify_employee" not in planned.required_tools


def test_invalid_employee_id(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("E-9999")
    assert [e.tool_name for e in result.tool_executions] == ["verify_employee"]
    assert result.tool_executions[0].status == ToolExecutionStatus.FAILED
    assert "couldn't be verified" in result.final_answer.lower()
    assert "vacation" not in result.final_answer.lower()
    mock_rag_service.query.assert_not_called()


def test_valid_employee_id(agent: AgentService) -> None:
    result = agent.run("E-1101")
    assert [e.tool_name for e in result.tool_executions] == ["verify_employee"]
    assert result.tool_executions[0].success
    assert "verified" in result.final_answer.lower()


def test_protected_tool_after_verification(agent: AgentService) -> None:
    result = agent.run(
        "How many vacation days do I have?",
        metadata={"employee_id": "E-1101"},
    )
    assert [e.tool_name for e in result.tool_executions] == ["get_leave_balance"]
    assert result.tool_executions[0].success
    assert "vacation" in result.final_answer.lower()


def test_leave_history_is_readable_not_raw_dict(agent: AgentService) -> None:
    result = agent.run(
        "Show my leave history",
        metadata={"employee_id": "E-1101"},
    )
    assert [e.tool_name for e in result.tool_executions] == ["get_leave_history"]
    assert result.tool_executions[0].success
    answer = result.final_answer
    assert "leave history" in answer.lower()
    assert "'leave_history'" not in answer
    assert '"leave_history"' not in answer
    assert "{" not in answer
    assert "·" in answer or "-" in answer or "|" in answer


def test_agent_metadata_exposes_execution_fields(agent: AgentService) -> None:
    result = agent.run(
        "How many vacation days do I have?",
        metadata={"employee_id": "E-1101", "api_key": "sk-secret-should-not-leak"},
    )
    meta = result.metadata
    assert meta.get("normalized_input") == "How many vacation days do I have?"
    assert meta.get("detected_intent") == IntentRoute.EMPLOYEE.value
    assert meta.get("selected_route") == IntentRoute.EMPLOYEE.value
    assert meta.get("verification_status") == "verified"
    assert meta.get("verified_employee_id") == "E-1101"
    assert meta.get("rag_used") is False
    assert isinstance(meta.get("tools_invoked"), list)
    assert meta["tools_invoked"][0]["tool_name"] == "get_leave_balance"
    assert meta["tools_invoked"][0]["result_summary"]
    assert "api_key" not in meta
    assert "sk-secret" not in str(meta)
    assert result.latency_ms is not None


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------


def test_policy_question_uses_rag(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("What is the vacation policy?")
    tools = [e.tool_name for e in result.tool_executions]
    assert tools == ["search_docs", "summarize"]
    assert mock_rag_service.query.called
    assert result.final_answer


def test_unrelated_question_does_not_return_random_rag_document(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("What is the capital of France?")
    _assert_no_tools(result, rag=mock_rag_service)
    lowered = result.final_answer.lower()
    assert "not sure" in lowered or "help with" in lowered
    assert "employee_handbook" not in lowered
    assert "paid leave" not in lowered


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


def test_hybrid_leave_policy_question(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run(
        "Can I carry forward my remaining vacation days?",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_leave_balance" in tools
    assert "search_company_policy" in tools
    assert mock_rag_service.query.called
    assert result.planner_output.intent_route == IntentRoute.HYBRID.value


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_empty_input(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("   ")
    _assert_no_tools(result, rag=mock_rag_service)
    assert result.final_answer


def test_unknown_input(agent: AgentService, mock_rag_service: MagicMock) -> None:
    result = agent.run("asdfghjkl")
    _assert_no_tools(result, rag=mock_rag_service)
    assert "not sure" in result.final_answer.lower() or "help with" in result.final_answer.lower()


def test_rag_failure(agent: AgentService, mock_rag_service: MagicMock) -> None:
    mock_rag_service.query.side_effect = RuntimeError("chroma collection missing")
    result = agent.run("What is the vacation policy?")
    assert any(e.tool_name == "search_docs" and not e.success for e in result.tool_executions)
    assert "OPENAI_API_KEY" not in result.final_answer
    assert "chroma" not in result.final_answer.lower()
    assert "trouble" in result.final_answer.lower() or "try again" in result.final_answer.lower()


def test_llm_generation_failure(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    def boom(request: Any) -> RAGResponse:
        raise RuntimeError("OpenAI authentication failed (check OPENAI_API_KEY)")

    mock_rag_service.query.side_effect = boom
    result = agent.run("What is the sick leave policy?")
    assert "OPENAI_API_KEY" not in result.final_answer
    assert "D:\\" not in result.final_answer
    assert result.final_answer


def test_tool_failure(agent: AgentService, employee_service: EmployeeService) -> None:
    tool = GetLeaveBalanceTool(employee_service)
    execution = tool.execute({}, context={})
    assert execution.status == ToolExecutionStatus.FAILED
    assert execution.error
    # No leave numbers leaked when unverified.
    assert execution.output == {} or "leave_balance" not in (execution.output or {})


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


def test_no_employee_data_before_verification(
    agent: AgentService, employee_service: EmployeeService
) -> None:
    planned = RuleBasedPlanner().plan("How many vacation days do I have?")
    assert planned.direct_answer
    assert all(name not in planned.execution_order for name in PROTECTED_EMPLOYEE_TOOLS)

    tool = GetLeaveBalanceTool(employee_service)
    execution = tool.execute(
        {"employee_id": "E-1101"},  # spoofed LLM arg must be ignored
        context={},
    )
    assert execution.status == ToolExecutionStatus.FAILED
    assert verified_employee_id_from_context({}) is None

    result = agent.run("How many vacation days do I have?")
    assert "vacation=" not in result.final_answer.lower()
    assert not any(
        e.tool_name in PROTECTED_EMPLOYEE_TOOLS and e.success
        for e in result.tool_executions
    )


def test_sanitize_strips_offline_footer() -> None:
    dirty = (
        "Paid leave accrues monthly.\n\n"
        "(Offline extractive answer from document "
        "`D:\\RAG-DEMO\\data\\sample\\employee_handbook.md::p0::c0` — "
        "set OPENAI_API_KEY for full LLM generation.)"
    )
    clean = sanitize_user_facing_answer(dirty)
    assert "OPENAI_API_KEY" not in clean
    assert "Offline extractive" not in clean
    assert "D:\\RAG-DEMO" not in clean
    assert "Paid leave" in clean


def test_offline_provider_hides_implementation_details() -> None:
    provider = OfflineExtractiveProvider()
    prompt = BuiltPrompt(
        system="Use context only.",
        user=(
            "Context:\n"
            "[Document 1] id=D:\\RAG-DEMO\\data\\sample\\employee_handbook.md::p0::c0 "
            "score=0.91 source=employee_handbook.md\n"
            "Employees accrue paid leave based on handbook policy.\n\n"
            "Question:\n"
            "What is the leave policy?\n\n"
            "Answer using only the context above."
        ),
        question="What is the leave policy?",
        context_document_count=1,
        context_char_length=60,
    )
    result = provider.generate(prompt)
    assert "OPENAI_API_KEY" not in result.answer
    assert "Offline extractive" not in result.answer
    assert "D:\\RAG-DEMO" not in result.answer
    assert "paid leave" in result.answer.lower()


def test_classify_intent_greeting() -> None:
    decision = classify_intent("hi")
    assert decision.route == IntentRoute.CONVERSATION
    assert decision.response


# ---------------------------------------------------------------------------
# Leave-balance tool-calling regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "So how many leaves are there?",
        "How many leaves do I have?",
        "What's my leave balance?",
        "How many vacation days are remaining?",
        "And how many leaves do I have?",
        "What about my remaining leaves?",
        "How much PTO do I have?",
        "How many days off can I take?",
    ],
)
def test_leave_balance_phrases_select_existing_tool(question: str) -> None:
    planned = RuleBasedPlanner().plan(
        question,
        metadata={"employee_id": "E-1101"},
    )
    assert planned.execution_order == ["get_leave_balance"]
    assert planned.direct_answer is None
    assert planned.intent_route == IntentRoute.EMPLOYEE.value


def test_holidays_then_leave_followup_routes_to_balance(agent: AgentService) -> None:
    holidays = agent.run(
        "What are the upcoming company holidays?",
        metadata={"employee_id": "E-1101"},
    )
    assert [e.tool_name for e in holidays.tool_executions] == [
        "get_upcoming_holidays"
    ]
    assert holidays.final_answer
    assert "not sure what you're asking" not in holidays.final_answer.lower()

    followup = agent.run(
        "So how many leaves are there?",
        metadata={
            "employee_id": "E-1101",
            "last_assistant_message": holidays.final_answer,
        },
    )
    assert [e.tool_name for e in followup.tool_executions] == ["get_leave_balance"]
    assert followup.tool_executions[0].success
    assert "not sure what you're asking" not in followup.final_answer.lower()
    assert "vacation" in followup.final_answer.lower()
    tools = followup.metadata.get("tools_invoked") or []
    assert tools
    assert tools[0]["tool_name"] == "get_leave_balance"
    assert tools[0]["status"] in {"success", "SUCCESS", "Success"}


def test_leave_balance_tools_called_metadata(agent: AgentService) -> None:
    result = agent.run(
        "How many leaves do I have?",
        metadata={"employee_id": "E-1101"},
    )
    assert [e.tool_name for e in result.tool_executions] == ["get_leave_balance"]
    tools = result.metadata.get("tools_invoked") or []
    assert len(tools) == 1
    assert tools[0]["tool_name"] == "get_leave_balance"
    assert "arguments" in tools[0]
    assert tools[0].get("result_summary")
    assert "not sure" not in result.final_answer.lower()


def test_leave_policy_still_uses_rag_not_balance(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("What is the leave policy?")
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_leave_balance" not in tools
    assert "search_docs" in tools

