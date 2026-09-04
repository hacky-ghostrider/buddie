"""Sprint 18B/C — deterministic DeepEval suite tests (no live LLM judge)."""

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
from evals.metrics.config import (
    METRIC_ANSWER_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_FINAL_RESPONSE_CORRECTNESS,
    METRIC_HALLUCINATION,
    BuddieDeepEvalConfig,
)
from evals.metrics.results import MetricScoreResult
from evals.runners.deepeval_case import DeepEvalCompatibleCase
from evals.runners.deepeval_suite import (
    evaluate_deepeval_case,
    format_suite_console,
    run_buddie_deepeval_suite,
)
from evals.runners.runtime_collector import build_retrieval_context, collect_deepeval_case


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
        correlation_id="rag-corr-s18bc",
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
def agent(mock_rag_service: MagicMock, tmp_path) -> AgentService:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    return AgentService(
        rag_service=mock_rag_service,
        employee_service=EmployeeService(store),
        tracing_service=TracingService(tracer=NoOpTracer()),
    )


@pytest.fixture(scope="module")
def dataset():
    return load_buddie_golden_dataset()


@pytest.fixture
def config() -> BuddieDeepEvalConfig:
    return BuddieDeepEvalConfig(
        faithfulness=0.7,
        answer_relevancy=0.7,
        hallucination=0.7,
        contextual_precision=0.7,
        contextual_recall=0.7,
        contextual_relevancy=0.7,
        final_response_correctness=0.7,
    )


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


def _failing_on_first_case_factory(fail_case_id: str):
    """Fail every metric for one case id; pass otherwise."""

    def _measure(
        metric_name: str,
        test_case: Any,
        *,
        threshold: float,
    ) -> MetricScoreResult:
        expected = getattr(test_case, "expected_output", "")
        if expected.startswith(f"FAIL::{fail_case_id}::"):
            return MetricScoreResult(
                name=metric_name,
                score=0.1,
                passed=False,
                threshold=threshold,
                reason="deterministic fail",
            )
        return MetricScoreResult(
            name=metric_name,
            score=0.95,
            passed=True,
            threshold=threshold,
            reason="deterministic pass",
        )

    return _measure


def test_golden_dataset_loads_thirty_six(dataset) -> None:
    assert len(dataset.cases) == 36


def test_deepeval_fields_populated(agent: AgentService, dataset) -> None:
    case = next(c for c in dataset.cases if c.id == "leave-balance-vacation-001")
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    kwargs = deepeval_case.to_llm_test_case_kwargs()
    assert kwargs["input"] == case.user_query
    assert kwargs["expected_output"] == case.expected_answer
    assert kwargs["actual_output"].strip()
    assert isinstance(kwargs["retrieval_context"], list)


def test_retrieval_context_runtime_only_never_expected_context(dataset) -> None:
    case = next(c for c in dataset.cases if c.expected_context)
    poisoned = case.model_copy(
        update={"expected_context": ["MUST_NOT_APPEAR_IN_RETRIEVAL"]}
    )

    class _Empty:
        evaluation_context = None
        tool_executions: list = []

    texts = build_retrieval_context(_Empty(), poisoned)  # type: ignore[arg-type]
    assert texts == []
    assert "MUST_NOT_APPEAR_IN_RETRIEVAL" not in texts


def test_expected_output_remains_golden_reference(
    agent: AgentService, dataset
) -> None:
    case = next(c for c in dataset.cases if c.id == "rag-carry-forward-cap-017")
    deepeval_case = collect_deepeval_case(dataset, case, agent)
    assert deepeval_case.expected_output == case.expected_answer
    assert deepeval_case.expected_context == list(case.expected_context)


def test_empty_retrieval_skips_context_metrics(config: BuddieDeepEvalConfig) -> None:
    case = DeepEvalCompatibleCase(
        case_id="neg-empty",
        input="Who is the CEO of Acme?",
        actual_output="I don't have that information in the available tools.",
        expected_output="Refuse — not in corpus",
        retrieval_context=[],
        expected_context=["gold reference only"],
        category="negative_unknown",
        expected_behavior="refuse_or_insufficient",
    )
    result = evaluate_deepeval_case(case, config, measure_fn=_passing_measure)
    assert result.faithfulness.skipped is True
    assert result.contextual_precision.skipped is True
    assert result.contextual_recall.skipped is True
    assert result.hallucination is not None and result.hallucination.skipped is True
    assert (
        result.contextual_relevancy is not None
        and result.contextual_relevancy.skipped is True
    )
    assert result.answer_relevancy.skipped is False
    assert result.answer_relevancy.passed is True
    assert result.final_response_correctness.passed is True
    assert "gold reference only" not in (case.retrieval_context or [])
    # Retrieval metrics N/A when expected_context empty; here expected has gold
    # but runtime retrieval is empty → zeros, not golden substitution.
    assert result.precision_at_1 == 0.0
    assert result.mrr == 0.0
    assert result.hit_at_5 == 0.0


def test_metric_results_schema(config: BuddieDeepEvalConfig) -> None:
    case = DeepEvalCompatibleCase(
        case_id="schema-001",
        input="How much vacation do I have?",
        actual_output="You have 14 vacation days remaining.",
        expected_output="14 vacation days remaining",
        retrieval_context=["get_leave_balance: {\"vacation\": 14}"],
        expected_context=["Employee has 14 vacation days remaining."],
        category="leave_hr",
        expected_behavior="answer_from_tool",
    )
    result = evaluate_deepeval_case(case, config, measure_fn=_passing_measure)
    assert result.case_id == "schema-001"
    assert result.category == "leave_hr"
    assert result.query == case.input
    assert result.overall_status == "passed"
    for name in (
        METRIC_FAITHFULNESS,
        METRIC_ANSWER_RELEVANCY,
        METRIC_HALLUCINATION,
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
        METRIC_FINAL_RESPONSE_CORRECTNESS,
    ):
        score = result.metric_map()[name]
        assert score.name == name
        assert score.score == 0.95
        assert score.passed is True
        assert score.threshold == 0.7
    flat = result.to_flat_metric_dict()
    assert {
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
    } <= set(flat)


def test_one_failed_case_does_not_stop_suite(
    agent: AgentService, dataset, config: BuddieDeepEvalConfig
) -> None:
    fail_id = "leave-balance-vacation-001"
    subset = [
        c
        for c in dataset.cases
        if c.id
        in {
            fail_id,
            "leave-balance-sick-002",
            "negative-external-ceo-024",
        }
    ]
    stamped = []
    for case in subset:
        if case.id == fail_id:
            stamped.append(
                case.model_copy(
                    update={
                        "expected_answer": f"FAIL::{fail_id}::{case.expected_answer}"
                    }
                )
            )
        else:
            stamped.append(case)

    mini = dataset.model_copy(update={"cases": stamped})
    report = run_buddie_deepeval_suite(
        agent,
        dataset=mini,
        config=config,
        measure_fn=_failing_on_first_case_factory(fail_id),
    )
    assert report.total_cases == 3
    assert fail_id in report.failed_case_ids
    assert report.failed >= 1
    assert report.passed + report.failed + report.errors + report.rate_limited == 3
    ids = {c.case_id for c in report.cases}
    assert ids == {c.id for c in stamped}
    console = format_suite_console(report)
    assert "Buddie Evaluation Suite" in console
    assert fail_id in console


def test_infrastructure_failure_continues(
    dataset, config: BuddieDeepEvalConfig
) -> None:
    class _Boom:
        def run(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("collector boom")

    subset = dataset.model_copy(update={"cases": list(dataset.cases[:3])})
    report = run_buddie_deepeval_suite(
        _Boom(),  # type: ignore[arg-type]
        dataset=subset,
        config=config,
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 3
    assert report.errors == 3
    assert report.error_case_ids == [c.id for c in subset.cases]
    for case in report.cases:
        assert case.overall_status == "error"
        assert case.infrastructure_error
        assert case.infrastructure_error.startswith("infrastructure:")


def test_full_suite_thirty_six_with_injected_metrics(
    agent: AgentService, dataset, config: BuddieDeepEvalConfig
) -> None:
    report = run_buddie_deepeval_suite(
        agent,
        dataset=dataset,
        config=config,
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 36
    assert report.adversarial_cases == 8
    assert report.errors == 0
    assert set(report.metric_averages) >= {
        METRIC_ANSWER_RELEVANCY,
        METRIC_FINAL_RESPONSE_CORRECTNESS,
    }
    payload = report.to_json_dict()
    assert payload["total_cases"] == 36
    assert len(payload["cases"]) == 36
    assert payload.get("annotation_summary")
    assert payload["annotation_summary"]["total_cases"] == 36
    first = payload["cases"][0]
    assert {
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
    } <= set(first)
