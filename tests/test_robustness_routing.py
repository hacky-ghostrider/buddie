"""Property-style robustness tests for Buddie conversational routing.

These assert category-level behavior (greeting / goodbye / gibberish / …),
not brittle exact-phrase dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.sanity

from app.agent.conversation import (
    ConversationKind,
    IntentRoute,
    UNKNOWN_FALLBACK,
    classify_intent,
    has_business_intent,
    looks_like_employee_id,
    normalize_for_routing,
)
from app.agent.service import AgentService
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
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


def _assert_no_rag_or_tools(result: Any, rag: MagicMock) -> None:
    assert result.tool_executions == []
    assert result.planner_output is not None
    assert result.planner_output.execution_order == []
    assert result.planner_output.direct_answer
    rag.query.assert_not_called()


GREETING_VARIANTS = [
    "hi",
    "hello",
    "hey",
    "hi!!!",
    "hey buddie",
    "good morning",
    "hello there",
    "helo",
    "hii",
    "gud morning",
]

GOODBYE_VARIANTS = [
    "bye",
    "goodbye",
    "cool bye",
    "okay bye",
    "thanks bye",
    "see you later",
    "catch you later",
    "bye!!!",
    "byee",
    "gotta go",
    "alright bye",
]

THANKS_VARIANTS = [
    "thanks",
    "thank you",
    "thanks buddie",
    "cool thanks",
    "okay thanks",
    "thx",
    "thanx",
    "thanks, that's helpful",
]

GIBBERISH_VARIANTS = [
    "asdfgh",
    "qwerty",
    "xyzabc",
    "blahblah",
    "!!!",
    "???",
    "...",
    "@@@",
]

NUMBER_VARIANTS = [
    "123",
    "12345",
    "2026",
    "1101",
    "0",
    "42",
]

EMPTY_VARIANTS = [
    "",
    " ",
    "    ",
    "\n",
    "\t",
    " \n\t ",
]

BUSINESS_CASUAL_VARIANTS = [
    "hey, how many vacation days do I have?",
    "hi buddie, what's my leave balance?",
    "cool, what's the vacation policy?",
    "thanks, what are the upcoming holidays?",
    "bye the way, how many vacation days do I have?",
    "cool, can I carry forward my leave?",
]


@pytest.mark.parametrize("text", EMPTY_VARIANTS)
def test_empty_whitespace_variants(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "help" in result.final_answer.lower()
    assert result.metadata.get("intent_route") == IntentRoute.EMPTY.value


@pytest.mark.parametrize("text", GREETING_VARIANTS)
def test_greeting_variants(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert result.metadata.get("intent_route") == IntentRoute.CONVERSATION.value
    decision = classify_intent(text)
    assert decision.kind == ConversationKind.GREETING


@pytest.mark.parametrize("text", GOODBYE_VARIANTS)
def test_goodbye_variants(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "bye" in result.final_answer.lower() or "day" in result.final_answer.lower()
    decision = classify_intent(text)
    assert decision.kind == ConversationKind.GOODBYE


@pytest.mark.parametrize("text", THANKS_VARIANTS)
def test_thanks_variants(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "welcome" in result.final_answer.lower()


@pytest.mark.parametrize("text", GIBBERISH_VARIANTS)
def test_gibberish_variants(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    lowered = result.final_answer.lower()
    assert "not sure" in lowered or "help with" in lowered
    assert "employee_handbook" not in lowered


@pytest.mark.parametrize("text", NUMBER_VARIANTS)
def test_numbers_do_not_verify_or_crash(
    agent: AgentService, mock_rag_service: MagicMock, text: str
) -> None:
    result = agent.run(text)
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert not looks_like_employee_id(text)
    # Bare digits stay on the graceful fallback path — no verify prompt.
    assert "not sure" in result.final_answer.lower() or "help with" in result.final_answer.lower()
    assert "please enter your employee id" not in result.final_answer.lower()
    assert "verify_employee" not in [
        e.tool_name for e in result.tool_executions
    ]
    assert result.metadata.get("verification_status") == "unverified"
    # Standalone numbers must not be treated as a successful verify route.
    assert result.metadata.get("selected_route") != IntentRoute.VERIFY_ID.value


@pytest.mark.parametrize(
    "text",
    [
        "E-1101",
        "employee ID is E-1101",
        "my employee id is E-1101",
    ],
)
def test_employee_id_phrases_enter_verify_flow(
    agent: AgentService, text: str
) -> None:
    result = agent.run(text)
    assert [e.tool_name for e in result.tool_executions] == ["verify_employee"]
    assert result.tool_executions[0].success
    assert result.planner_output.intent_route == IntentRoute.VERIFY_ID.value


@pytest.mark.parametrize("text", BUSINESS_CASUAL_VARIANTS)
def test_business_intent_preserved_with_casual_wrappers(text: str) -> None:
    assert has_business_intent(text)
    decision = classify_intent(text)
    assert decision.route == IntentRoute.PASS_THROUGH


def test_casual_prefix_employee_routes_to_verification(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("hey, how many vacation days do I have?")
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "verify" in result.final_answer.lower()
    assert result.planner_output.intent_route == IntentRoute.EMPLOYEE.value


def test_casual_prefix_policy_uses_rag(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run("cool, what's the vacation policy?")
    tools = [e.tool_name for e in result.tool_executions]
    assert tools == ["search_docs", "summarize"]
    assert mock_rag_service.query.called


def test_hybrid_with_casual_prefix(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run(
        "cool, can I carry forward my leave?",
        metadata={"employee_id": "E-1101"},
    )
    tools = [e.tool_name for e in result.tool_executions]
    assert "get_leave_balance" in tools
    assert "search_company_policy" in tools
    assert result.planner_output.intent_route == IntentRoute.HYBRID.value


def test_short_ack_with_context_is_thanks(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run(
        "cool",
        metadata={
            "last_assistant_message": "You have 14 vacation days remaining.",
        },
    )
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "welcome" in result.final_answer.lower()


def test_okay_bye_after_answer(
    agent: AgentService, mock_rag_service: MagicMock
) -> None:
    result = agent.run(
        "okay bye",
        metadata={
            "last_assistant_message": "You have 14 vacation days remaining.",
        },
    )
    _assert_no_rag_or_tools(result, mock_rag_service)
    assert "bye" in result.final_answer.lower()


def test_explicit_employee_id_still_verifies(agent: AgentService) -> None:
    result = agent.run("E-1101")
    assert [e.tool_name for e in result.tool_executions] == ["verify_employee"]
    assert result.tool_executions[0].success


def test_normalize_for_routing_is_tolerant() -> None:
    assert normalize_for_routing("Hi!!!") == "hi"
    assert normalize_for_routing("cool, bye") == "cool bye"
    assert "bye" in normalize_for_routing("byee")
    decision = classify_intent("1101")
    assert decision.route == IntentRoute.UNKNOWN
    assert decision.response == UNKNOWN_FALLBACK
    assert "please enter your employee id" not in (decision.response or "").lower()


def test_real_business_questions_still_pass_through() -> None:
    for text in (
        "what is the vacation policy?",
        "how many vacation days do I have?",
        "show my leave history",
        "what are the upcoming holidays?",
        "Can I carry forward my leave?",
    ):
        assert classify_intent(text).route == IntentRoute.PASS_THROUGH


def test_extract_employee_id_from_phrases() -> None:
    from app.agent.conversation import extract_employee_id

    assert extract_employee_id("E-1101") == "E-1101"
    assert extract_employee_id("employee ID is E-1101") == "E-1101"
    assert extract_employee_id("my employee id is E-1101") == "E-1101"
    assert extract_employee_id("123") is None
    assert extract_employee_id("1101") is None
    assert extract_employee_id("asdfgh") is None
    # Business questions that cite an id should not become verify-only.
    assert extract_employee_id("Show leave history for E-1101") is None
