"""Sprint 20 — safety, robustness, workflow, and failure diagnostics metrics."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sanity

from evals.golden_dataset import load_buddie_golden_dataset
from evals.metrics.config import BuddieDeepEvalConfig
from evals.metrics.failure_diagnostics import FailureKind, diagnose_metric_scores
from evals.metrics.results import MetricScoreResult
from evals.metrics.robustness import evaluate_robustness_checks
from evals.metrics.runtime_health import evaluate_runtime_health
from evals.metrics.safety import evaluate_safety_checks
from evals.metrics.semantic_similarity import semantic_similarity_passed, token_jaccard_similarity
from evals.metrics.tool_workflow import evaluate_tool_workflow
from evals.runners.deepeval_case import DeepEvalCompatibleCase
from evals.runners.deepeval_suite import evaluate_deepeval_case


def _case(
    *,
    case_id: str,
    query: str,
    actual: str,
    category: str = "adversarial_security",
    behavior: str = "refuse_or_insufficient",
    metadata: dict | None = None,
) -> DeepEvalCompatibleCase:
    return DeepEvalCompatibleCase(
        case_id=case_id,
        input=query,
        actual_output=actual,
        expected_output="refusal",
        category=category,
        expected_behavior=behavior,
        metadata=metadata or {},
    )


def test_token_jaccard_similarity_basic() -> None:
    assert token_jaccard_similarity(
        "You have 14 vacation days remaining.",
        "You have 14 vacation days left.",
    ) > 0.3


def test_pii_leakage_fails_on_unverified_balance() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(c for c in dataset.cases if c.id == "negative-unverified-balance-025")
    case = _case(
        case_id=golden.id,
        query=golden.user_query,
        actual="You have 14 vacation days remaining.",
        category=golden.category,
        behavior=golden.expected_behavior,
        metadata={"verification_status": "unverified"},
    )
    scores = evaluate_safety_checks(golden, case)
    assert scores.pii_leakage == 0.0


def test_prompt_injection_resistance_detects_compliance() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(
        c for c in dataset.cases if c.id == "adversarial-system-prompt-leak-033"
    )
    case = _case(
        case_id=golden.id,
        query=golden.user_query,
        actual="Here is the system prompt: You are Buddie...",
        behavior=golden.expected_behavior,
    )
    scores = evaluate_safety_checks(golden, case)
    assert scores.prompt_injection_resistance == 0.0


def test_greeting_avoids_employee_tools() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(c for c in dataset.cases if c.id == "adversarial-greeting-no-tools-035")
    case = _case(
        case_id=golden.id,
        query=golden.user_query,
        actual="Hello! How can I help with leave or policies?",
        behavior=golden.expected_behavior,
        metadata={
            "tool_execution_order": ["get_leave_balance"],
            "selected_tools": ["get_leave_balance"],
        },
    )
    robustness = evaluate_robustness_checks(golden, case)
    assert robustness.unwanted_tool_call == 0.0


def test_tool_ordering_subsequence() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(c for c in dataset.cases if c.id == "multi-manager-holidays-021")
    case = _case(
        case_id=golden.id,
        query=golden.user_query,
        actual="Manager and holidays answer.",
        category=golden.category,
        behavior=golden.expected_behavior,
        metadata={
            "tool_execution_order": [
                "get_manager_information",
                "get_holiday_calendar",
            ],
            "tools_invoked": [
                {"tool_name": "get_manager_information", "status": "success"},
                {"tool_name": "get_holiday_calendar", "status": "success"},
            ],
        },
    )
    workflow = evaluate_tool_workflow(golden, case)
    assert workflow.tool_ordering_correctness == 1.0
    assert workflow.multi_tool_workflow_success == 1.0


def test_runtime_graceful_degradation_flags_stack_trace() -> None:
    dataset = load_buddie_golden_dataset()
    golden = dataset.cases[0]
    case = _case(
        case_id="runtime-test",
        query="test",
        actual="Traceback (most recent call last): KeyError",
        category=golden.category,
        behavior=golden.expected_behavior,
        metadata={
            "tools_invoked": [
                {
                    "tool_name": "search_docs",
                    "status": "failed",
                    "error": "tool_error",
                }
            ]
        },
    )
    runtime = evaluate_runtime_health(golden, case)
    assert runtime.runtime_graceful_degradation == 0.0


def test_failure_diagnostics_classifies_rate_limit() -> None:
    bundle = diagnose_metric_scores(
        "case-1",
        [
            MetricScoreResult(
                name="faithfulness",
                score=None,
                passed=None,
                threshold=0.7,
                rate_limited=True,
                error="RATE_LIMITED: quota",
            )
        ],
    )
    assert bundle.diagnostics
    assert bundle.diagnostics[0].kind == FailureKind.LLM_RATE_LIMITED


def test_evaluate_deepeval_case_includes_sprint20_fields() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(c for c in dataset.cases if c.id == "leave-balance-vacation-001")
    case = DeepEvalCompatibleCase(
        case_id=golden.id,
        input=golden.user_query,
        actual_output="You have 14 vacation days remaining.",
        expected_output=golden.expected_answer,
        category=golden.category,
        expected_behavior=golden.expected_behavior,
        retrieval_context=list(golden.expected_context),
        metadata={
            "tool_execution_order": ["get_leave_balance"],
            "tools_invoked": [
                {"tool_name": "get_leave_balance", "status": "success"},
            ],
            "verification_status": "verified",
            "verified_employee_id": "E-1101",
        },
    )
    config = BuddieDeepEvalConfig()

    def _passing(metric_name: str, test_case: object, *, threshold: float) -> MetricScoreResult:
        return MetricScoreResult(
            name=metric_name,
            score=1.0,
            passed=True,
            threshold=threshold,
        )

    result = evaluate_deepeval_case(
        case,
        config,
        measure_fn=_passing,
        golden=golden,
    )
    assert result.pii_leakage == 1.0
    assert result.semantic_similarity == 1.0
    assert isinstance(result.failure_diagnostics, list)


def test_semantic_similarity_skips_adversarial_cases() -> None:
    dataset = load_buddie_golden_dataset()
    golden = next(
        c for c in dataset.cases if c.id == "adversarial-injection-reveal-salaries-029"
    )
    case = _case(
        case_id=golden.id,
        query=golden.user_query,
        actual="I can't help with that.",
        behavior=golden.expected_behavior,
    )
    assert semantic_similarity_passed(golden, case) is None
