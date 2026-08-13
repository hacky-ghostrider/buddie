"""Sprint 18 phase 1 — golden → Buddie runtime → DeepEval-compatible cases.

Deterministic: mocked RAG, seeded employee store, no live DeepEval metrics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.agent.service import AgentService
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService
from evals.golden_dataset import (
    BUDDIE_GOLDEN_CASES_PATH,
    load_buddie_golden_dataset,
)
from evals.runners import (
    DeepEvalCompatibleCase,
    collect_all_deepeval_cases,
    collect_deepeval_case,
    session_metadata_for_case,
)
from evals.runners.runtime_collector import build_retrieval_context


def _rag_response(question: str, answer: str) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_documents=[
            RetrievedDocument(
                id="chunk-handbook-1",
                text=(
                    "Carry-forward: unused vacation may carry forward up to 5 days. "
                    "Sick leave and personal leave do not carry forward. "
                    "Requests over 10 vacation days typically need manager approval."
                ),
                metadata={"source": "employee_handbook.md"},
                score=0.93,
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
        correlation_id="rag-corr-s18",
    )


@pytest.fixture
def mock_rag_service() -> MagicMock:
    service = MagicMock()
    service.query.side_effect = lambda request: _rag_response(
        request.question,
        "Handbook-grounded answer from the employee policy corpus.",
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


@pytest.fixture(scope="module")
def dataset():
    return load_buddie_golden_dataset()


def test_buddie_golden_loader_reads_baseline(dataset) -> None:
    assert BUDDIE_GOLDEN_CASES_PATH.is_file()
    assert dataset.name == "buddie_golden_cases"
    assert len(dataset.cases) == 28
    assert dataset.default_session.get("verified_employee_id") == "E-1101"


def test_session_metadata_verified_by_default(dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "leave-balance-vacation-001")
    meta = session_metadata_for_case(dataset, case)
    assert meta["verified_employee_id"] == "E-1101"
    assert meta["evaluation_case_id"] == case.id


def test_session_metadata_omits_verification_for_negative(dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "negative-unverified-balance-025")
    meta = session_metadata_for_case(dataset, case)
    assert "verified_employee_id" not in meta
    assert meta["evaluation_behavior"] == "require_verification"


def test_collect_single_tool_case_maps_deepeval_fields(agent: AgentService, dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "leave-balance-vacation-001")
    deepeval_case = collect_deepeval_case(dataset, case, agent)

    assert isinstance(deepeval_case, DeepEvalCompatibleCase)
    assert deepeval_case.case_id == case.id
    assert deepeval_case.input == case.user_query
    assert deepeval_case.expected_output == case.expected_answer
    assert deepeval_case.actual_output.strip()
    assert deepeval_case.actual_output != "(empty)"
    assert deepeval_case.retrieval_context
    assert any("get_leave_balance" in item for item in deepeval_case.retrieval_context)
    assert "14" in deepeval_case.actual_output or "vacation" in deepeval_case.actual_output.lower()

    kwargs = deepeval_case.to_llm_test_case_kwargs()
    assert set(kwargs) >= {
        "input",
        "actual_output",
        "expected_output",
        "retrieval_context",
        "context",
    }
    assert kwargs["input"] == case.user_query
    assert kwargs["expected_output"] == case.expected_answer


def test_collect_unverified_case_does_not_leak_balance(
    agent: AgentService, dataset
) -> None:
    case = next(c for c in dataset.cases if c.id == "negative-unverified-balance-025")
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    assert deepeval_case.metadata.get("verification_status") == "unverified"
    # Must not invent Avery's vacation balance when unverified.
    assert "14" not in deepeval_case.actual_output


def test_collect_hitl_case_keeps_confirmation_behavior(
    agent: AgentService, dataset
) -> None:
    case = next(c for c in dataset.cases if c.id == "multi-leave-request-hitl-023")
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    tools = deepeval_case.metadata.get("tool_execution_order") or []
    assert "create_leave_request" not in tools
    assert deepeval_case.actual_output.strip()


def test_collect_rag_case_includes_retrieval_chunks(
    agent: AgentService, dataset, mock_rag_service: MagicMock
) -> None:
    case = next(c for c in dataset.cases if c.id == "rag-carry-forward-cap-017")
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    assert mock_rag_service.query.called
    assert deepeval_case.retrieval_context
    joined = " ".join(deepeval_case.retrieval_context).lower()
    assert "carry" in joined or "vacation" in joined


def test_build_retrieval_context_never_substitutes_expected_context(dataset) -> None:
    """Golden expected_context must stay reference-only, even when runtime is empty."""
    case = next(c for c in dataset.cases if c.expected_context)
    poisoned = case.model_copy(update={"expected_context": ["gold-only-context"]})

    class _Empty:
        evaluation_context = None
        tool_executions: list = []

    texts = build_retrieval_context(_Empty(), poisoned)  # type: ignore[arg-type]
    assert texts == []
    assert "gold-only-context" not in texts
    assert poisoned.expected_context == ["gold-only-context"]


def test_build_retrieval_context_empty_when_no_runtime_evidence(dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "negative-external-ceo-024")

    class _Empty:
        evaluation_context = None
        tool_executions: list = []

    assert build_retrieval_context(_Empty(), case) == []  # type: ignore[arg-type]


def test_collect_preserves_expected_output_and_separates_contexts(
    agent: AgentService, dataset
) -> None:
    case = next(c for c in dataset.cases if c.expected_context)
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    assert deepeval_case.expected_output == case.expected_answer
    assert deepeval_case.expected_context == list(case.expected_context)
    assert deepeval_case.expected_context is not deepeval_case.retrieval_context
    assert isinstance(deepeval_case.retrieval_context, list)


def test_collect_all_twenty_eight_cases(agent: AgentService, dataset) -> None:
    cases = collect_all_deepeval_cases(dataset, agent)
    assert len(cases) == 28
    ids = {c.case_id for c in cases}
    assert ids == {c.id for c in dataset.cases}
    for item in cases:
        assert item.input.strip()
        assert item.actual_output.strip()
        assert item.expected_output.strip()
        assert isinstance(item.retrieval_context, list)
        kwargs = item.to_llm_test_case_kwargs()
        assert kwargs["actual_output"] == item.actual_output
