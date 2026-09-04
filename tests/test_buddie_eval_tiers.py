"""CI tier runs — smoke/sanity golden subsets wired to Buddie eval metrics.

Deterministic (mocked agent + injected measure_fn). No live Gemini judge.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.agent.service import AgentService
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService
from evals.golden_dataset import case_ids_for_tier, load_buddie_golden_dataset
from evals.metrics.config import BuddieDeepEvalConfig
from evals.metrics.results import MetricScoreResult
from evals.runners.deepeval_suite import run_buddie_deepeval_suite


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
        correlation_id="rag-corr-tier",
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


@pytest.fixture
def config() -> BuddieDeepEvalConfig:
    return BuddieDeepEvalConfig()


@pytest.mark.smoke
def test_smoke_tier_golden_count() -> None:
    dataset = load_buddie_golden_dataset()
    assert len(case_ids_for_tier(dataset, "smoke")) == 4


@pytest.mark.smoke
def test_smoke_tier_suite_runs_with_metrics(
    agent: AgentService,
    config: BuddieDeepEvalConfig,
) -> None:
    report = run_buddie_deepeval_suite(
        agent,
        config=config,
        test_tier="smoke",
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 4
    assert report.errors == 0
    assert report.passed + report.failed + report.rate_limited == 4
    first = report.cases[0]
    assert isinstance(first.failure_diagnostics, list)
    flat = first.to_flat_metric_dict()
    assert {
        "faithfulness",
        "tool_correctness",
        "task_completion",
        "semantic_similarity",
    } <= set(flat)


@pytest.mark.sanity
def test_sanity_tier_golden_count() -> None:
    dataset = load_buddie_golden_dataset()
    assert len(case_ids_for_tier(dataset, "sanity")) == 12


@pytest.mark.sanity
def test_sanity_tier_suite_runs_with_safety_metrics(
    agent: AgentService,
    config: BuddieDeepEvalConfig,
) -> None:
    report = run_buddie_deepeval_suite(
        agent,
        config=config,
        test_tier="sanity",
        measure_fn=_passing_measure,
    )
    assert report.total_cases == 12
    assert report.errors == 0
    assert report.passed + report.failed + report.rate_limited == 12

    injection = next(
        c for c in report.cases if c.case_id == "adversarial-injection-reveal-salaries-029"
    )
    flat = injection.to_flat_metric_dict()
    assert flat.get("prompt_injection_resistance") is not None
    assert flat.get("pii_leakage") is not None
    assert flat.get("adversarial_refusal") is not None
