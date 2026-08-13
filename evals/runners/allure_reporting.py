"""Allure attachment helpers for Buddie evaluation cases.

Reporting/UI only — does not replace DeepEval, retrieval metrics, or JSON.
"""

from __future__ import annotations

import json
from typing import Any

from evals.metrics.results import CaseEvaluationResult, MetricScoreResult

_DEEPEVAL_METRIC_ATTRS = (
    "faithfulness",
    "answer_relevancy",
    "hallucination",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "final_response_correctness",
)

_FLAT_METRIC_KEYS = (
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
    "tool_correctness",
    "argument_correctness",
    "hitl_correctness",
    "task_completion",
)


def _metric_summary(metric: MetricScoreResult | None) -> str:
    if metric is None:
        return "n/a"
    if metric.skipped:
        return f"skipped ({metric.skip_reason or 'n/a'})"
    if metric.rate_limited:
        return f"RATE_LIMITED: {metric.error or 'Gemini quota exceeded'}"
    if metric.error:
        return f"LLM_EVAL_ERROR: {metric.error}"
    return (
        f"score={metric.score} passed={metric.passed} "
        f"threshold={metric.threshold}"
    )


def attach_case_to_allure(case: CaseEvaluationResult) -> None:
    """Attach Buddie case fields/metrics to the current Allure test result."""
    import allure

    allure.dynamic.parameter("Case ID", case.case_id)
    if case.category:
        allure.dynamic.parameter("Category", case.category)
        allure.dynamic.tag(case.category)
    allure.dynamic.parameter("Overall status", case.overall_status)

    allure.attach(
        case.query,
        name="User query",
        attachment_type=allure.attachment_type.TEXT,
    )
    if case.expected_output is not None:
        allure.attach(
            case.expected_output,
            name="Expected answer",
            attachment_type=allure.attachment_type.TEXT,
        )
    if case.actual_output is not None:
        allure.attach(
            case.actual_output,
            name="Actual answer",
            attachment_type=allure.attachment_type.TEXT,
        )
    allure.attach(
        json.dumps(case.expected_tools, indent=2),
        name="Expected tool(s)",
        attachment_type=allure.attachment_type.JSON,
    )
    allure.attach(
        json.dumps(case.actual_tools, indent=2),
        name="Actual tool(s)",
        attachment_type=allure.attachment_type.JSON,
    )

    metric_rows: dict[str, Any] = {}
    for attr in _DEEPEVAL_METRIC_ATTRS:
        metric = getattr(case, attr, None)
        label = attr.replace("_", " ").title()
        if isinstance(metric, MetricScoreResult):
            summary = _metric_summary(metric)
            allure.dynamic.parameter(label, summary)
            metric_rows[attr] = metric.to_public_dict()
        else:
            allure.dynamic.parameter(label, "n/a")
            metric_rows[attr] = None

    flat = case.to_flat_metric_dict()
    for key in _FLAT_METRIC_KEYS:
        value = flat.get(key)
        label = key.replace("_", " ").title() if not key.startswith(("precision", "recall", "hit", "mrr")) else key
        # Keep Precision@K / Recall@K / Hit@K / MRR readable.
        pretty = {
            "precision_at_1": "Precision@1",
            "precision_at_3": "Precision@3",
            "precision_at_5": "Precision@5",
            "recall_at_1": "Recall@1",
            "recall_at_3": "Recall@3",
            "recall_at_5": "Recall@5",
            "hit_at_1": "Hit@1",
            "hit_at_3": "Hit@3",
            "hit_at_5": "Hit@5",
            "mrr": "MRR",
            "tool_correctness": "Tool correctness",
            "argument_correctness": "Argument correctness",
            "hitl_correctness": "HITL correctness",
            "task_completion": "Task completion",
        }.get(key, label)
        allure.dynamic.parameter(
            pretty,
            "n/a" if value is None else str(value),
        )
        metric_rows[key] = value

    if case.failure_reasons:
        allure.attach(
            "\n".join(case.failure_reasons),
            name="Failure reason",
            attachment_type=allure.attachment_type.TEXT,
        )
    else:
        allure.dynamic.parameter("Failure reason", "none")

    allure.attach(
        json.dumps(metric_rows, indent=2, default=str),
        name="Metric scores (JSON)",
        attachment_type=allure.attachment_type.JSON,
    )

    if case.overall_status == "error":
        allure.dynamic.label("failure_kind", "llm_or_infrastructure_error")
    elif case.overall_status == "rate_limited":
        allure.dynamic.label("failure_kind", "rate_limited")
    elif case.overall_status == "failed":
        allure.dynamic.label("failure_kind", "metric_or_agent_assertion")
    else:
        allure.dynamic.label("failure_kind", "none")


def assert_case_evaluation(case: CaseEvaluationResult) -> None:
    """Fail the current test using existing suite status (no new thresholds)."""
    if case.overall_status == "passed":
        return
    reasons = "; ".join(case.failure_reasons) or case.overall_status
    if case.overall_status == "rate_limited":
        raise AssertionError(
            f"Gemini rate limit for {case.case_id}: {reasons}"
        )
    if case.overall_status == "error":
        raise AssertionError(
            f"LLM/evaluation error for {case.case_id}: {reasons}"
        )
    raise AssertionError(
        f"Metric/assertion failure for {case.case_id}: {reasons}"
    )


__all__ = [
    "assert_case_evaluation",
    "attach_case_to_allure",
]
