"""Allure reporting for the Buddie golden evaluation cases.

Each golden case is one pytest/Allure test. DeepEval + retrieval + agent
checks remain the evaluation layer; Allure is reporting/UI only.

By default this uses the same deterministic ``measure_fn`` pattern as other
Buddie eval tests (no live LLM). To render a live JSON report into Allure:

    set BUDDIE_EVAL_REPORT=data/reports/buddie_eval_suite.json
    pytest tests/test_buddie_eval_allure.py --alluredir=data/reports/allure/latest

Optional tier filter (smoke / sanity / regression):

    set BUDDIE_EVAL_TIER=sanity
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.regression

from app.agent.service import AgentService
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService
from evals.golden_dataset import case_ids_for_tier, load_buddie_golden_dataset
from evals.metrics.config import BuddieDeepEvalConfig
from evals.metrics.results import MetricScoreResult, SuiteEvaluationReport
from evals.runners.allure_reporting import (
    assert_case_evaluation,
    attach_case_to_allure,
)
from evals.runners.deepeval_suite import run_buddie_deepeval_suite

pytest.importorskip("allure")


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
        correlation_id="rag-corr-allure",
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


def _selected_tier() -> str | None:
    tier = os.getenv("BUDDIE_EVAL_TIER", "").strip().lower()
    if tier in {"smoke", "sanity", "regression"}:
        return tier
    return None


@pytest.fixture(scope="module")
def dataset():
    return load_buddie_golden_dataset()


@pytest.fixture(scope="module")
def case_ids(dataset) -> list[str]:
    tier = _selected_tier()
    if tier is not None:
        return case_ids_for_tier(dataset, tier)  # type: ignore[arg-type]
    return [case.id for case in dataset.cases]


@pytest.fixture(scope="module")
def agent(tmp_path_factory) -> AgentService:
    mock_rag = MagicMock()
    mock_rag.query.side_effect = lambda request: _rag_response(
        request.question,
        "Handbook-grounded answer from the employee policy corpus.",
    )
    store_dir = tmp_path_factory.mktemp("employees-allure")
    store = EmployeeStore(store_dir / "employees.json")
    store.seed()
    return AgentService(
        rag_service=mock_rag,
        employee_service=EmployeeService(store),
        tracing_service=TracingService(tracer=NoOpTracer()),
    )


@pytest.fixture(scope="module")
def suite_report(agent: AgentService, dataset) -> SuiteEvaluationReport:
    """Load a prior JSON report when ``BUDDIE_EVAL_REPORT`` is set; else run suite."""
    report_path = os.getenv("BUDDIE_EVAL_REPORT", "").strip()
    if report_path:
        path = Path(report_path)
        return SuiteEvaluationReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    config = BuddieDeepEvalConfig()
    tier = _selected_tier()
    return run_buddie_deepeval_suite(
        agent,
        dataset=dataset,
        config=config,
        measure_fn=_passing_measure,
        test_tier=tier,
        include_annotation_summary=True,
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_id" not in metafunc.fixturenames:
        return
    data = load_buddie_golden_dataset()
    tier = _selected_tier()
    if tier is not None:
        ids = case_ids_for_tier(data, tier)  # type: ignore[arg-type]
    else:
        ids = [case.id for case in data.cases]
    metafunc.parametrize("case_id", ids, ids=ids)


def test_buddie_eval_case_allure(
    case_id: str,
    suite_report: SuiteEvaluationReport,
) -> None:
    """One Allure test per golden case; continue collecting results on failure."""
    case = next((c for c in suite_report.cases if c.case_id == case_id), None)
    assert case is not None, f"Missing evaluation result for {case_id}"
    attach_case_to_allure(case)
    assert_case_evaluation(case)
