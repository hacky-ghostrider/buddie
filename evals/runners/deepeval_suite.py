"""Buddie evaluation suite — collect runtime cases, score, continue on failure.

Sprint 19 complete layer:
    28 goldens → AgentService → DeepEvalCompatibleCase
      → DeepEval generation/contextual metrics
      → deterministic retrieval metrics (expected_context vs retrieval_context)
      → deterministic agent checks
      → suite report + annotation summary

A single failed golden case does not abort the remaining cases.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from evals.golden_dataset.loader import load_buddie_golden_dataset
from evals.golden_dataset.models import BuddieGoldenCase, BuddieGoldenDataset
from evals.metrics.agent_checks import evaluate_agent_checks
from evals.metrics.annotations import build_annotation_report
from evals.metrics.config import (
    METRIC_ANSWER_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_FINAL_RESPONSE_CORRECTNESS,
    METRIC_HALLUCINATION,
    BuddieDeepEvalConfig,
    default_buddie_deepeval_config,
)
from evals.metrics.g_eval import measure_final_response_correctness
from evals.metrics.results import (
    CaseEvaluationResult,
    MetricScoreResult,
    SuiteEvaluationReport,
)
from evals.metrics.retrieval import compute_retrieval_metrics
from evals.metrics.standard import MetricMeasureFn, measure_all_standard_metrics
from evals.runners.deepeval_case import DeepEvalCompatibleCase
from evals.runners.runtime_collector import AgentRunner, collect_deepeval_case

logger = logging.getLogger(__name__)

CollectFn = Callable[
    [BuddieGoldenDataset, BuddieGoldenCase, AgentRunner],
    DeepEvalCompatibleCase,
]

_DEEPEVAL_METRIC_NAMES = (
    METRIC_FAITHFULNESS,
    METRIC_ANSWER_RELEVANCY,
    METRIC_HALLUCINATION,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FINAL_RESPONSE_CORRECTNESS,
)


def _placeholder_metric(name: str, threshold: float, error: str) -> MetricScoreResult:
    return MetricScoreResult(
        name=name,
        score=None,
        passed=False,
        threshold=threshold,
        error=error,
    )


def _overall_status(
    scores: Sequence[MetricScoreResult],
    *,
    agent_failure_reasons: Sequence[str],
    infrastructure_error: str | None,
) -> tuple[str, list[str]]:
    if infrastructure_error:
        return "error", [infrastructure_error]

    reasons: list[str] = []
    rate_limited_metrics = False
    llm_eval_errors = False
    metric_threshold_failures = False
    for score in scores:
        if score.skipped:
            continue
        if score.rate_limited:
            rate_limited_metrics = True
            reasons.append(f"{score.name}: RATE_LIMITED: {score.error}")
        elif score.error:
            llm_eval_errors = True
            reasons.append(f"{score.name}: LLM_EVAL_ERROR: {score.error}")
        elif score.passed is False:
            metric_threshold_failures = True
            detail = score.reason or f"score={score.score} < {score.threshold}"
            reasons.append(f"{score.name}: {detail}")
    reasons.extend(agent_failure_reasons)

    if not reasons:
        return "passed", []
    # Genuine threshold / agent assertion failures take precedence.
    if metric_threshold_failures or agent_failure_reasons:
        return "failed", reasons
    if rate_limited_metrics and not llm_eval_errors:
        return "rate_limited", reasons
    if llm_eval_errors:
        return "error", reasons
    return "failed", reasons


def evaluate_deepeval_case(
    case: DeepEvalCompatibleCase,
    config: BuddieDeepEvalConfig,
    *,
    measure_fn: MetricMeasureFn | None = None,
    golden: BuddieGoldenCase | None = None,
) -> CaseEvaluationResult:
    """Score one case with DeepEval + retrieval + optional agent checks."""
    standard = measure_all_standard_metrics(
        case, config, measure_fn=measure_fn, golden=golden
    )
    g_eval = measure_final_response_correctness(
        case, config, measure_fn=measure_fn
    )
    scores = [
        standard[METRIC_FAITHFULNESS],
        standard[METRIC_ANSWER_RELEVANCY],
        standard[METRIC_HALLUCINATION],
        standard[METRIC_CONTEXTUAL_PRECISION],
        standard[METRIC_CONTEXTUAL_RECALL],
        standard[METRIC_CONTEXTUAL_RELEVANCY],
        g_eval,
    ]

    expected_context = list(
        golden.expected_context if golden is not None else case.expected_context
    )
    retrieval = compute_retrieval_metrics(case.retrieval_context, expected_context)

    from evals.metrics.agent_checks import (
        actual_tools_from_case,
        expected_tools_from_golden,
    )

    agent_reasons: list[str] = []
    tool_c = arg_c = hitl_c = task_c = None
    expected_tools: list[str] = []
    actual_tools = actual_tools_from_case(case)
    if golden is not None:
        expected_tools = expected_tools_from_golden(golden)
        agent = evaluate_agent_checks(golden, case)
        tool_c = agent.tool_correctness
        arg_c = agent.argument_correctness
        hitl_c = agent.hitl_correctness
        task_c = agent.task_completion
        agent_reasons = agent.failure_reasons()

    status, reasons = _overall_status(
        scores,
        agent_failure_reasons=agent_reasons,
        infrastructure_error=None,
    )
    return CaseEvaluationResult(
        case_id=case.case_id,
        category=case.category,
        query=case.input,
        expected_behavior=case.expected_behavior,
        faithfulness=standard[METRIC_FAITHFULNESS],
        answer_relevancy=standard[METRIC_ANSWER_RELEVANCY],
        hallucination=standard[METRIC_HALLUCINATION],
        contextual_precision=standard[METRIC_CONTEXTUAL_PRECISION],
        contextual_recall=standard[METRIC_CONTEXTUAL_RECALL],
        contextual_relevancy=standard[METRIC_CONTEXTUAL_RELEVANCY],
        final_response_correctness=g_eval,
        precision_at_1=retrieval.precision_at_1,
        precision_at_3=retrieval.precision_at_3,
        precision_at_5=retrieval.precision_at_5,
        recall_at_1=retrieval.recall_at_1,
        recall_at_3=retrieval.recall_at_3,
        recall_at_5=retrieval.recall_at_5,
        hit_at_1=retrieval.hit_at_1,
        hit_at_3=retrieval.hit_at_3,
        hit_at_5=retrieval.hit_at_5,
        mrr=retrieval.mrr,
        tool_correctness=tool_c,
        argument_correctness=arg_c,
        hitl_correctness=hitl_c,
        task_completion=task_c,
        overall_status=status,  # type: ignore[arg-type]
        failure_reasons=reasons,
        retrieval_context_count=len(case.retrieval_context),
        expected_output=case.expected_output,
        actual_output=case.actual_output,
        expected_tools=expected_tools,
        actual_tools=actual_tools,
    )


def _infrastructure_case_result(
    *,
    case_id: str,
    category: str | None,
    query: str,
    expected_behavior: str | None,
    config: BuddieDeepEvalConfig,
    error: str,
) -> CaseEvaluationResult:
    err = f"infrastructure: {error}"
    placeholders = {
        name: _placeholder_metric(name, config.threshold_for(name), err)
        for name in _DEEPEVAL_METRIC_NAMES
    }
    return CaseEvaluationResult(
        case_id=case_id,
        category=category,
        query=query,
        expected_behavior=expected_behavior,
        faithfulness=placeholders[METRIC_FAITHFULNESS],
        answer_relevancy=placeholders[METRIC_ANSWER_RELEVANCY],
        hallucination=placeholders[METRIC_HALLUCINATION],
        contextual_precision=placeholders[METRIC_CONTEXTUAL_PRECISION],
        contextual_recall=placeholders[METRIC_CONTEXTUAL_RECALL],
        contextual_relevancy=placeholders[METRIC_CONTEXTUAL_RELEVANCY],
        final_response_correctness=placeholders[
            METRIC_FINAL_RESPONSE_CORRECTNESS
        ],
        overall_status="error",
        failure_reasons=[err],
        infrastructure_error=err,
    )


def _metric_averages(
    cases: Sequence[CaseEvaluationResult],
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for case in cases:
        for name, score in case.metric_map().items():
            if score.skipped or score.score is None:
                continue
            buckets.setdefault(name, []).append(float(score.score))
        for name, value in case.to_flat_metric_dict().items():
            if value is None:
                continue
            # Avoid double-counting DeepEval nested scores already added.
            if name in case.metric_map():
                continue
            buckets.setdefault(name, []).append(float(value))
    return {
        name: round(sum(values) / len(values), 6)
        for name, values in buckets.items()
        if values
    }


def build_suite_report(
    case_results: Sequence[CaseEvaluationResult],
    config: BuddieDeepEvalConfig,
    *,
    notes: str | None = None,
    annotation_summary: dict[str, Any] | None = None,
) -> SuiteEvaluationReport:
    """Aggregate per-case results into a suite report."""
    passed = sum(1 for c in case_results if c.overall_status == "passed")
    failed = sum(1 for c in case_results if c.overall_status == "failed")
    errors = sum(1 for c in case_results if c.overall_status == "error")
    rate_limited = sum(
        1 for c in case_results if c.overall_status == "rate_limited"
    )
    return SuiteEvaluationReport(
        total_cases=len(case_results),
        passed=passed,
        failed=failed,
        errors=errors,
        rate_limited=rate_limited,
        metric_averages=_metric_averages(case_results),
        failed_case_ids=[
            c.case_id for c in case_results if c.overall_status == "failed"
        ],
        error_case_ids=[
            c.case_id for c in case_results if c.overall_status == "error"
        ],
        rate_limited_case_ids=[
            c.case_id for c in case_results if c.overall_status == "rate_limited"
        ],
        failure_reasons_by_case={
            c.case_id: list(c.failure_reasons)
            for c in case_results
            if c.failure_reasons
        },
        cases=list(case_results),
        thresholds={
            name: config.threshold_for(name) for name in _DEEPEVAL_METRIC_NAMES
        },
        annotation_summary=annotation_summary,
        notes=notes,
    )


def run_buddie_deepeval_suite(
    agent: AgentRunner,
    *,
    dataset: BuddieGoldenDataset | None = None,
    config: BuddieDeepEvalConfig | None = None,
    measure_fn: MetricMeasureFn | None = None,
    collect_fn: CollectFn | None = None,
    validate_tools: bool = False,
    case_ids: Sequence[str] | None = None,
    include_annotation_summary: bool = True,
) -> SuiteEvaluationReport:
    """Execute and score Buddie golden cases without aborting on single failures."""
    cfg = config or default_buddie_deepeval_config()
    data = dataset or load_buddie_golden_dataset()
    collector = collect_fn or (
        lambda ds, golden, runner: collect_deepeval_case(
            ds, golden, runner, validate_tools=validate_tools
        )
    )

    selected = list(data.cases)
    if case_ids is not None:
        wanted = set(case_ids)
        selected = [c for c in data.cases if c.id in wanted]

    results: list[CaseEvaluationResult] = []
    for golden in selected:
        logger.info("Evaluating Buddie case: %s", golden.id)
        try:
            deepeval_case = collector(data, golden, agent)
            result = evaluate_deepeval_case(
                deepeval_case,
                cfg,
                measure_fn=measure_fn,
                golden=golden,
            )
        except Exception as exc:  # noqa: BLE001 — continue suite
            logger.exception(
                "Infrastructure failure collecting/evaluating case=%s",
                golden.id,
            )
            result = _infrastructure_case_result(
                case_id=golden.id,
                category=golden.category,
                query=golden.user_query,
                expected_behavior=golden.expected_behavior,
                config=cfg,
                error=str(exc),
            )
        results.append(result)

    annotation = None
    if include_annotation_summary:
        annotation = build_annotation_report(data).to_public_dict()

    return build_suite_report(
        results,
        cfg,
        annotation_summary=annotation,
    )


# Alias used by Sprint 19 docs / CLI.
run_buddie_eval_suite = run_buddie_deepeval_suite


def format_suite_console(report: SuiteEvaluationReport) -> str:
    """Concise human-readable console summary."""
    lines = [
        "Buddie Evaluation Suite",
        f"  total={report.total_cases} passed={report.passed} "
        f"failed={report.failed} rate_limited={report.rate_limited} "
        f"errors={report.errors}",
        "  metric averages:",
    ]
    if report.metric_averages:
        for name, value in sorted(report.metric_averages.items()):
            lines.append(f"    {name}: {value:.4f}")
    else:
        lines.append("    (none)")

    if report.failed_case_ids:
        lines.append(f"  failed case ids: {', '.join(report.failed_case_ids)}")
    if report.rate_limited_case_ids:
        lines.append(
            f"  rate_limited case ids: {', '.join(report.rate_limited_case_ids)}"
        )
    if report.error_case_ids:
        lines.append(f"  error case ids: {', '.join(report.error_case_ids)}")
    if report.failure_reasons_by_case:
        lines.append("  failure reasons:")
        for case_id, reasons in report.failure_reasons_by_case.items():
            for reason in reasons:
                lines.append(f"    {case_id}: {reason}")

    lines.append("  cases:")
    for case in report.cases:
        flat = case.to_flat_metric_dict()
        lines.append(
            f"    [{case.overall_status}] {case.case_id} "
            f"({case.category or '-'})"
        )
        for name, score in case.metric_map().items():
            if score.skipped:
                detail = f"skipped ({score.skip_reason})"
            elif score.rate_limited:
                detail = f"RATE_LIMITED {score.error}"
            elif score.error:
                detail = f"ERROR {score.error}"
            elif score.passed is True:
                detail = f"{score.score} pass=True"
            else:
                detail = f"{score.score} pass={score.passed}"
            lines.append(f"      {name}: {detail}")
        for key in (
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
        ):
            value = flat.get(key)
            if value is not None:
                lines.append(f"      {key}: {value}")
        if case.failure_reasons:
            for reason in case.failure_reasons:
                lines.append(f"      reason: {reason}")
    return "\n".join(lines)


def write_suite_report_json(
    report: SuiteEvaluationReport,
    path: str | Any,
) -> None:
    """Write machine-readable JSON report to ``path``."""
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


__all__ = [
    "build_suite_report",
    "evaluate_deepeval_case",
    "format_suite_console",
    "run_buddie_deepeval_suite",
    "run_buddie_eval_suite",
    "write_suite_report_json",
]
