"""Sprint 10.2 — Agent evaluation preparation tests (no live APIs).

Covers EvaluationContext, ToolContract, ToolExecution, ToolTraceMapper,
and the canonical agent-tools-foundation-001 dataset. LangSmith is mocked
via plain dict payloads — never a live Client call.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.evaluation.context import EvaluationContext
from app.evaluation.dataset import GoldenDatasetLoader
from app.evaluation.scenarios import (
    CANONICAL_DATASET_PATH,
    CANONICAL_SCENARIO_ID,
    SPRINT_10_2_ACTUAL_TOOLS,
    SPRINT_11_EXPECTED_ACTUAL_TOOLS,
)
from app.evaluation.tool_validation import (
    ToolContract,
    ToolExecution,
    ToolTraceMapper,
    ToolValidator,
    contracts_from_golden_fields,
)
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument


def _doc() -> RetrievedDocument:
    return RetrievedDocument(
        id="c1",
        text="Leave policy excerpt from the employee handbook.",
        metadata={"file_name": "employee_handbook.pdf"},
        score=0.91,
    )


def _rag() -> RAGResponse:
    return RAGResponse(
        question="Summarize the leave policy from the employee handbook.",
        answer=(
            "The employee handbook leave policy defines paid time off "
            "eligibility, accrual, approval workflow, and notice requirements."
        ),
        retrieved_documents=[_doc()],
        retrieval_metadata={},
        generation_metadata={
            "model": "gpt-4o-mini",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt": {"system": "sys", "user": "Summarize leave policy"},
            "prompt_version": "rag_user@v1",
        },
        latency=LatencyBreakdown(
            retrieval_ms=5.0,
            prompt_build_ms=1.0,
            llm_ms=20.0,
            total_ms=26.0,
        ),
        correlation_id="corr-10.2",
    )


# ---------------------------------------------------------------------------
# EvaluationContext
# ---------------------------------------------------------------------------


def test_evaluation_context_from_rag_response_populates_core_fields() -> None:
    ctx = EvaluationContext.from_rag_response(
        question="Summarize the leave policy from the employee handbook.",
        rag_response=_rag(),
        expected_answer="Leave policy summary",
        expected_sources=["employee_handbook.pdf"],
        langsmith_run_id="run-1",
        langsmith_trace_id="trace-1",
        langsmith_run_url="https://smith.langchain.com/runs/run-1",
        cost_usd=0.001,
        metadata={"suite": "canonical"},
    )

    assert ctx.question.startswith("Summarize the leave policy")
    assert ctx.original_user_request == ctx.question
    assert ctx.answer
    assert ctx.generated_answer == ctx.answer
    assert len(ctx.retrieved_documents) == 1
    assert ctx.retrieved_chunks[0].startswith("Leave policy")
    assert ctx.model == "gpt-4o-mini"
    assert ctx.prompt_version == "rag_user@v1"
    assert ctx.token_usage["total_tokens"] == 150
    assert ctx.latency_ms == pytest.approx(26.0)
    assert ctx.cost_usd == pytest.approx(0.001)
    assert ctx.langsmith_run_id == "run-1"
    assert ctx.langsmith_trace_id == "trace-1"
    assert ctx.correlation_id == "corr-10.2"
    assert ctx.tool_calls == []
    assert ctx.metadata["suite"] == "canonical"


def test_evaluation_context_rejects_blank_question() -> None:
    with pytest.raises(Exception):
        EvaluationContext(question="   ", answer="x")


def test_evaluation_context_metric_backward_compat_constructor() -> None:
    """Sprint 9 metrics construct context with question/answer/docs only."""
    ctx = EvaluationContext(
        question="What is RAG?",
        answer="Retrieval-augmented generation.",
        retrieved_documents=[],
    )
    assert ctx.answer == "Retrieval-augmented generation."
    assert ctx.generated_answer == ctx.answer
    assert ctx.tool_calls == []


# ---------------------------------------------------------------------------
# ToolContract / ToolExecution
# ---------------------------------------------------------------------------


def test_tool_contract_validates_required_and_expected_args() -> None:
    contract = ToolContract(
        tool_name="search_docs",
        required=["query"],
        optional=["top_k"],
        expected_arguments={"query": "leave policy"},
        expected_execution_order=0,
        minimum_calls=1,
        maximum_calls=1,
        maximum_latency_ms=500.0,
        expected_output_type="list",
    )
    assert contract.validate_arguments({"query": "leave policy", "top_k": 3}) == []
    failures = contract.validate_arguments({"top_k": 3})
    assert any("query" in f for f in failures)

    expectation = contract.to_expectation()
    assert expectation.tool_name == "search_docs"
    assert expectation.order == 0
    assert expectation.max_latency_ms == 500.0


def test_tool_contract_custom_argument_validator() -> None:
    contract = ToolContract(
        tool_name="search_docs",
        required=["query"],
        argument_validators={"query": lambda v: isinstance(v, str) and len(v) > 3},
    )
    assert contract.validate_arguments({"query": "leave"}) == []
    assert contract.validate_arguments({"query": "ab"})


def test_tool_execution_success_property() -> None:
    from app.evaluation.tool_validation.tool_execution import ToolExecutionStatus

    ok = ToolExecution(
        tool_name="summarize",
        arguments={"document": "employee_handbook.pdf"},
        output="Leave policy summary",
        status=ToolExecutionStatus.SUCCESS,
        latency_ms=12.0,
        retry_count=0,
        order=1,
    )
    assert ok.success is True
    bad = ok.model_copy(
        update={"status": ToolExecutionStatus.FAILED, "error": "timeout"}
    )
    assert bad.success is False
    # Legacy string aliases still coerce via model_validate.
    legacy = ToolExecution.model_validate(
        {**ok.model_dump(mode="json"), "status": "error", "error": "boom"}
    )
    assert legacy.status == ToolExecutionStatus.FAILED


def test_contracts_from_golden_fields_preserve_order() -> None:
    contracts = contracts_from_golden_fields(
        expected_tool_order=["search_docs", "summarize"],
        expected_tool_arguments=[
            {"query": "leave policy employee handbook"},
            {"document": "employee_handbook.pdf"},
        ],
    )
    assert [c.tool_name for c in contracts] == ["search_docs", "summarize"]
    assert contracts[0].expected_execution_order == 0
    assert contracts[1].expected_arguments["document"] == "employee_handbook.pdf"


# ---------------------------------------------------------------------------
# ToolTraceMapper (mocked LangSmith payloads)
# ---------------------------------------------------------------------------


def test_tool_trace_mapper_maps_langsmith_child_runs() -> None:
    started = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 8, 7, 10, 0, 0, 500000, tzinfo=timezone.utc)
    trace = {
        "child_runs": [
            {
                "name": "search_docs",
                "run_type": "tool",
                "inputs": {"query": "leave policy employee handbook"},
                "outputs": [{"id": "doc-1"}],
                "start_time": started.isoformat(),
                "end_time": finished.isoformat(),
                "id": "child-1",
            },
            {
                "tool_name": "summarize",
                "arguments": {"document": "employee_handbook.pdf"},
                "output": "summary text",
                "latency_ms": 18.0,
                "status": "success",
                "retry_count": 1,
            },
        ]
    }

    mapper = ToolTraceMapper()
    executions = mapper.map_langsmith_trace(trace)

    assert [e.tool_name for e in executions] == ["search_docs", "summarize"]
    assert executions[0].arguments["query"] == "leave policy employee handbook"
    from app.evaluation.tool_validation.tool_execution import ToolExecutionStatus

    assert executions[0].status == ToolExecutionStatus.SUCCESS
    assert executions[0].latency_ms == pytest.approx(500.0)
    assert executions[1].retry_count == 1
    assert executions[1].order == 1

    actual = mapper.to_actual_tool_calls(executions)
    assert actual[0].tool_name == "search_docs"
    assert actual[0].success is True
    assert actual[1].metadata["retry_count"] == 1


def test_tool_trace_mapper_empty_trace_yields_no_tools() -> None:
    mapper = ToolTraceMapper()
    assert mapper.map_langsmith_trace(None) == []
    assert mapper.map_langsmith_trace({}) == []
    assert mapper.map_to_actual_tool_calls({"tool_calls": []}) == []


def test_tool_validator_uses_mapped_executions_via_contracts() -> None:
    contracts = contracts_from_golden_fields(
        expected_tool_order=["search_docs", "summarize"],
        expected_tool_arguments=[
            {"query": "leave policy employee handbook"},
            {"document": "employee_handbook.pdf"},
        ],
    )
    # Sprint 10.2: no agent → empty executions → validation fails expectedly
    report = ToolValidator().validate_contracts(contracts, [])
    assert report.passed is False
    assert report.expected_tools == ["search_docs", "summarize"]
    assert report.actual_tools == []

    # Future Sprint 11 shape (still unit-tested with mocks, no LangGraph)
    executions = ToolTraceMapper().map_langsmith_trace(
        {
            "tool_calls": [
                {
                    "tool_name": "search_docs",
                    "arguments": {"query": "leave policy employee handbook"},
                    "output": ["hit"],
                    "status": "success",
                    "latency_ms": 10.0,
                },
                {
                    "tool_name": "summarize",
                    "arguments": {"document": "employee_handbook.pdf"},
                    "output": "summary",
                    "status": "success",
                    "latency_ms": 12.0,
                },
            ]
        }
    )
    future = ToolValidator().validate_contracts(contracts, executions)
    assert future.passed is True
    assert future.actual_tools == SPRINT_11_EXPECTED_ACTUAL_TOOLS


# ---------------------------------------------------------------------------
# Canonical dataset / LangSmith reference scenario
# ---------------------------------------------------------------------------


def test_canonical_dataset_agent_tools_foundation_001() -> None:
    path = Path(CANONICAL_DATASET_PATH)
    assert path.is_file()
    examples = GoldenDatasetLoader().load(path)
    assert len(examples) == 1
    example = examples[0]
    assert example.id == CANONICAL_SCENARIO_ID
    assert example.question == (
        "Summarize the leave policy from the employee handbook."
    )
    assert example.expected_sources == ["employee_handbook.pdf"]
    assert example.expected_tools == ["search_docs", "summarize"]
    assert example.expected_tool_order == ["search_docs", "summarize"]
    assert example.expected_tool_arguments[0]["query"]
    assert example.expected_tool_arguments[1]["document"] == "employee_handbook.pdf"
    assert SPRINT_10_2_ACTUAL_TOOLS == []
    assert example.metadata.get("sprint_10_2_actual_tools") == []
    assert example.metadata.get("sprint_11_actual_tools") == [
        "search_docs",
        "summarize",
    ]


def test_golden_dataset_embeds_canonical_scenario() -> None:
    examples = GoldenDatasetLoader().load("datasets/golden_dataset.json")
    match = [e for e in examples if e.id == CANONICAL_SCENARIO_ID]
    assert len(match) == 1
    assert match[0].expected_tools == ["search_docs", "summarize"]
