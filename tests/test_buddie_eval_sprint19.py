"""Sprint 19 — retrieval metrics, annotations, agent checks, eval output."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.sanity

from app.agent.service import AgentService
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService
from evals.golden_dataset import load_buddie_golden_dataset
from evals.metrics.agent_checks import (
    evaluate_agent_checks,
    hitl_correctness_score,
    tool_correctness_score,
)
from evals.metrics.annotations import (
    EXPECTED_CATEGORY_COUNTS,
    build_annotation_report,
    format_annotation_console,
)
from evals.metrics.config import BuddieDeepEvalConfig
from evals.metrics.results import MetricScoreResult
from evals.metrics.retrieval import (
    compute_retrieval_metrics,
    hit_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from evals.runners.deepeval_case import DeepEvalCompatibleCase
from evals.runners.deepeval_suite import run_buddie_eval_suite


def _rag_response(question: str, answer: str) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_documents=[
            RetrievedDocument(
                id="chunk-handbook-1",
                text=(
                    "Carry-forward: unused vacation may carry forward up to 5 days. "
                    "Sick leave and personal leave do not carry forward."
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
        correlation_id="rag-corr-s19",
    )


@pytest.fixture
def agent(tmp_path) -> AgentService:
    mock_rag = MagicMock()
    mock_rag.query.side_effect = lambda request: _rag_response(
        request.question,
        "Handbook-grounded answer from the employee policy corpus.",
    )
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    return AgentService(
        rag_service=mock_rag,
        employee_service=EmployeeService(store),
        tracing_service=TracingService(tracer=NoOpTracer()),
    )


@pytest.fixture(scope="module")
def dataset():
    return load_buddie_golden_dataset()


@pytest.fixture
def config() -> BuddieDeepEvalConfig:
    return BuddieDeepEvalConfig()


def _passing_measure(
    metric_name: str,
    test_case: Any,
    *,
    threshold: float,
) -> MetricScoreResult:
    del test_case
    return MetricScoreResult(
        name=metric_name,
        score=0.95,
        passed=True,
        threshold=threshold,
        reason="deterministic pass",
    )


def test_annotation_integrity_thirty_six(dataset) -> None:
    report = build_annotation_report(dataset)
    assert report.total_cases == 36
    assert report.by_category["adversarial_security"] == 8
    assert len(report.cases_with_expected_answer) == 36
    assert report.counts["expected_context"] >= 1
    assert report.counts["expected_tool"] >= 1
    assert report.counts["hitl_expectation"] >= 1
    assert report.counts["negative_unknown_behavior"] >= 1
    console = format_annotation_console(report)
    assert "Total cases: 36" in console
    assert "leave_hr: 8" in console
    assert "HITL expectation:" in console


def test_runtime_evidence_separation_in_dataset_model(dataset) -> None:
    case = next(c for c in dataset.cases if c.expected_context)
    # Annotated fields exist on golden; runtime fields are separate.
    assert case.expected_answer
    assert isinstance(case.expected_context, list)
    assert case.user_query
    # DeepEvalCompatibleCase keeps expected_context off retrieval_context.
    deepeval = DeepEvalCompatibleCase(
        case_id=case.id,
        input=case.user_query,
        actual_output="runtime answer",
        expected_output=case.expected_answer,
        retrieval_context=[],
        expected_context=list(case.expected_context),
    )
    assert deepeval.retrieval_context == []
    assert deepeval.expected_context == list(case.expected_context)
    assert deepeval.expected_context is not deepeval.retrieval_context


def test_precision_recall_hit_mrr_calculations() -> None:
    expected = [
        "employee_id=E-1101 leave_balance vacation=14 sick=8 personal=3",
        "Unused vacation may carry forward up to 5 days",
    ]
    retrieved = [
        "get_leave_balance: {\"employee_id\": \"E-1101\", \"vacation\": 14, "
        "\"sick\": 8, \"personal\": 3}",
        "unrelated noise about cafeteria menus",
        "handbook: unused vacation may carry forward up to 5 days; sick leave "
        "does not carry forward",
    ]
    assert precision_at_k(retrieved, expected, 1) == 1.0
    assert precision_at_k(retrieved, expected, 3) == pytest.approx(2 / 3, rel=1e-3)
    assert recall_at_k(retrieved, expected, 1) == pytest.approx(0.5)
    assert recall_at_k(retrieved, expected, 3) == 1.0
    assert hit_at_k(retrieved, expected, 1) == 1.0
    assert hit_at_k(["noise only"], expected, 5) == 0.0
    assert mean_reciprocal_rank(retrieved, expected) == 1.0
    assert mean_reciprocal_rank(["noise", retrieved[0]], expected) == 0.5

    empty_expected = compute_retrieval_metrics(retrieved, [])
    assert empty_expected.mrr is None
    assert empty_expected.precision_at_5 is None

    zeroed = compute_retrieval_metrics([], expected)
    assert zeroed.precision_at_1 == 0.0
    assert zeroed.recall_at_5 == 0.0
    assert zeroed.hit_at_3 == 0.0
    assert zeroed.mrr == 0.0


def test_tool_and_hitl_correctness(dataset) -> None:
    leave = next(c for c in dataset.cases if c.id == "leave-balance-vacation-001")
    hitl = next(c for c in dataset.cases if c.id == "multi-leave-request-hitl-023")
    unverified = next(
        c for c in dataset.cases if c.id == "negative-unverified-balance-025"
    )

    good_leave = DeepEvalCompatibleCase(
        case_id=leave.id,
        input=leave.user_query,
        actual_output="You have 14 vacation days remaining.",
        expected_output=leave.expected_answer,
        retrieval_context=["get_leave_balance: {\"vacation\": 14}"],
        expected_behavior=leave.expected_behavior,
        metadata={
            "tool_execution_order": ["get_leave_balance"],
            "selected_tools": ["get_leave_balance"],
            "awaiting_confirmation": False,
            "verification_status": "verified",
            "tools_invoked": [
                {
                    "tool_name": "get_leave_balance",
                    "arguments": {"employee_id": "E-1101"},
                }
            ],
        },
    )
    assert tool_correctness_score(leave, good_leave) == 1.0
    scores = evaluate_agent_checks(leave, good_leave)
    assert scores.argument_correctness == 1.0
    assert scores.task_completion == 1.0

    bad_hitl = DeepEvalCompatibleCase(
        case_id=hitl.id,
        input=hitl.user_query,
        actual_output="Leave created.",
        expected_output=hitl.expected_answer,
        retrieval_context=[],
        expected_behavior=hitl.expected_behavior,
        metadata={
            "tool_execution_order": [
                "get_employee_profile",
                "get_leave_balance",
                "check_leave_eligibility",
                "create_leave_request",
            ],
            "awaiting_confirmation": False,
        },
    )
    assert hitl_correctness_score(hitl, bad_hitl) == 0.0
    assert tool_correctness_score(hitl, bad_hitl) == 0.0

    good_hitl = DeepEvalCompatibleCase(
        case_id=hitl.id,
        input=hitl.user_query,
        actual_output="Please confirm to submit your leave request.",
        expected_output=hitl.expected_answer,
        retrieval_context=["check_leave_eligibility: eligible=true"],
        expected_behavior=hitl.expected_behavior,
        metadata={
            "tool_execution_order": [
                "get_employee_profile",
                "get_leave_balance",
                "check_leave_eligibility",
            ],
            "awaiting_confirmation": True,
            "verification_status": "verified",
        },
    )
    assert hitl_correctness_score(hitl, good_hitl) == 1.0
    assert tool_correctness_score(hitl, good_hitl) == 1.0

    unverified_case = DeepEvalCompatibleCase(
        case_id=unverified.id,
        input=unverified.user_query,
        actual_output="Please enter your employee ID, for example E-1101.",
        expected_output=unverified.expected_answer,
        retrieval_context=[],
        expected_behavior=unverified.expected_behavior,
        metadata={
            "tool_execution_order": [],
            "verification_status": "unverified",
        },
    )
    assert tool_correctness_score(unverified, unverified_case) == 1.0
    assert evaluate_agent_checks(unverified, unverified_case).task_completion == 1.0


def test_failure_isolation_continues_all_cases(
    agent: AgentService, dataset, config: BuddieDeepEvalConfig
) -> None:
    boom_ids = {"leave-balance-vacation-001"}

    def _collect(ds, golden, runner):
        if golden.id in boom_ids:
            raise RuntimeError("isolated boom")
        from evals.runners.runtime_collector import collect_deepeval_case

        return collect_deepeval_case(ds, golden, runner)

    report = run_buddie_eval_suite(
        agent,
        dataset=dataset,
        config=config,
        measure_fn=_passing_measure,
        collect_fn=_collect,
    )
    assert report.total_cases == 36
    assert "leave-balance-vacation-001" in report.error_case_ids
    assert len(report.cases) == 36
    assert report.errors >= 1
    # Other cases still evaluated
    assert any(c.overall_status != "error" for c in report.cases)


def test_final_evaluation_output_schema(
    agent: AgentService, dataset, config: BuddieDeepEvalConfig
) -> None:
    subset = dataset.model_copy(update={"cases": list(dataset.cases[:5])})
    report = run_buddie_eval_suite(
        agent,
        dataset=subset,
        config=config,
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 5
    assert report.annotation_summary is not None
    payload = report.to_json_dict()
    for case in payload["cases"]:
        for key in (
            "case_id",
            "category",
            "query",
            "faithfulness",
            "answer_relevancy",
            "hallucination",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "mrr",
            "final_response_correctness",
            "tool_correctness",
            "argument_correctness",
            "hitl_correctness",
            "task_completion",
            "overall_status",
        ):
            assert key in case


def test_twenty_eight_case_execution_completes(
    agent: AgentService, dataset, config: BuddieDeepEvalConfig
) -> None:
    report = run_buddie_eval_suite(
        agent,
        dataset=dataset,
        config=config,
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 36
    assert report.errors == 0
    assert len(report.cases) == 36
    assert {c.case_id for c in report.cases} == {c.id for c in dataset.cases}
    # Every case has flat metric keys present (values may be null).
    for case in report.cases:
        flat = case.to_flat_metric_dict()
        assert flat["final_response_correctness"] == 0.95
        assert "tool_correctness" in flat
        assert "mrr" in flat
