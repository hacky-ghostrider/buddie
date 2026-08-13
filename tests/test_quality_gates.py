"""Sprint 12 — Continuous AI Evaluation / Quality Gates tests.

All DeepEval, LangSmith, and Agent dependencies are mocked.
No live API calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from app.agent.models import ExecutionStrategy, PlannerDecision, PlannerOutput
from app.config.settings import Settings
from app.evaluation.benchmark.dashboard import BenchmarkDashboardModel
from app.evaluation.benchmark.history import BenchmarkHistory, BenchmarkRunRecord
from app.evaluation.benchmark.runner import BenchmarkRunner, BenchmarkSummary
from app.evaluation.continuous import ContinuousEvaluationService
from app.evaluation.models import MetricResult
from app.evaluation.quality import (
    QualityGate,
    QualityGateEngine,
    QualityGateThresholds,
    QualityStatus,
    build_default_rules,
)
from app.evaluation.quality.decision import QualityDecision
from app.evaluation.quality.exceptions import QualityGateDisabledError
from app.evaluation.quality.models import QualityRuleId
from app.evaluation.quality.recommendations import recommendations_for_results
from app.evaluation.quality.report import QualityReportWriter
from app.evaluation.regression import CostRegression, RegressionRunner
from app.evaluation.report import EvaluationReport
from app.evaluation.timeline import EvaluationTimeline, TimelineStage
from app.evaluation.tool_validation.report import ToolValidationReport
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionMetrics,
    ToolExecutionStatus,
)
from app.evaluation.tool_validation.tool_result import (
    CalculatorResultData,
    ToolResult,
)
from app.retrieval.models import RetrievedDocument


def _settings(**overrides: Any) -> Settings:
    base = dict(
        app_env="test",
        enable_evaluation=True,
        enable_deepeval=False,
        enable_langsmith=False,
        enable_tool_validation=True,
        quality_gate_enabled=True,
        min_faithfulness=0.7,
        max_hallucination=0.3,
        min_relevancy=0.7,
        min_context_precision=0.6,
        min_context_recall=0.6,
        max_tool_failures=0,
        max_tool_latency=60_000.0,
        max_total_latency=120_000.0,
        max_cost=1.0,
        quality_pass_threshold=0.7,
        warning_threshold=0.6,
        default_pass_threshold=0.7,
        metric_timeout=5.0,
        report_directory="./data/reports-test",
        benchmark_directory="./data/benchmarks-test",
        quality_report_directory="./data/quality_reports-test",
        benchmark_history_path="./data/benchmarks-test/history.json",
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


def _metric(name: str, score: float, *, passed: bool | None = None) -> MetricResult:
    # Platform convention: all MetricResult scores are higher-is-better
    # (DeepEval hallucination is inverted in the adapter).
    threshold_ok = score >= 0.7
    return MetricResult(
        name=name,
        score=score,
        passed=threshold_ok if passed is None else passed,
    )


def _report(
    *,
    question: str = "What is leave policy?",
    overall_via_metrics: list[MetricResult] | None = None,
    pass_threshold: float = 0.7,
    latency: float = 100.0,
    cost: float | None = 0.01,
    tool_passed: bool | None = True,
    answer: str = "Employees receive 20 days of annual leave.",
) -> EvaluationReport:
    # Keep mean overall_score >= 0.7; hallucination is higher-is-better (0.9 ≈ raw 0.1).
    metrics = overall_via_metrics or [
        _metric("faithfulness", 0.95),
        _metric("hallucination", 0.9),
        _metric("answer_relevancy", 0.95),
        _metric("contextual_precision", 0.9),
        _metric("contextual_recall", 0.9),
    ]
    tool = None
    if tool_passed is not None:
        tool = ToolValidationReport(
            passed=tool_passed,
            expected_tools=["search_docs", "summarize"],
            actual_tools=["search_docs", "summarize"],
            failures=[] if tool_passed else ["tool mismatch"],
            matches=[],
        )
    return EvaluationReport.build(
        question=question,
        answer=answer,
        retrieved_documents=[
            RetrievedDocument(
                id="c1",
                text="Leave policy: 20 days.",
                score=0.9,
                metadata={},
            )
        ],
        metrics=metrics,
        latency_ms=latency,
        pass_threshold=pass_threshold,
        rag_latency_ms=latency,
        estimated_cost_usd=cost,
        langsmith_run_url="https://smith.langchain.com/public/demo",
        tool_validation=tool,
        metadata={"correlation_id": "corr-test-1"},
    )


class TestQualityGateEngine:
    def test_pass_when_all_rules_satisfied(self) -> None:
        settings = _settings()
        engine = QualityGateEngine(settings=settings)
        decision = engine.evaluate(_report(), correlation_id="c1")
        assert decision.status == QualityStatus.PASS
        assert decision.failed_rules == []
        assert decision.correlation_id == "c1"
        assert decision.overall_score >= 0.7

    def test_fail_on_low_faithfulness(self) -> None:
        engine = QualityGateEngine(settings=_settings(min_faithfulness=0.95))
        report = _report(
            overall_via_metrics=[
                _metric("faithfulness", 0.5),
                _metric("hallucination", 0.9),
                _metric("answer_relevancy", 0.9),
            ]
        )
        decision = engine.evaluate(report)
        assert decision.status == QualityStatus.FAIL
        assert QualityRuleId.MIN_FAITHFULNESS.value in decision.failed_rules
        assert decision.recommendations
        assert any("Faithfulness" in r.message for r in decision.recommendations)

    def test_fail_on_high_hallucination(self) -> None:
        engine = QualityGateEngine(settings=_settings(max_hallucination=0.2))
        report = _report(
            overall_via_metrics=[
                _metric("faithfulness", 0.9),
                # Higher-is-better 0.1 ⇒ raw hallucination 0.9 > 0.2 ceiling
                _metric("hallucination", 0.1),
                _metric("answer_relevancy", 0.9),
            ]
        )
        decision = engine.evaluate(report)
        assert decision.status == QualityStatus.FAIL
        assert QualityRuleId.MAX_HALLUCINATION.value in decision.failed_rules

    def test_warning_on_context_precision(self) -> None:
        # Lower overall pass threshold so only the WARNING-severity precision rule fires.
        engine = QualityGateEngine(
            settings=_settings(
                min_context_precision=0.9,
                quality_pass_threshold=0.5,
                warning_threshold=0.4,
            )
        )
        report = _report(
            overall_via_metrics=[
                _metric("faithfulness", 0.95),
                _metric("hallucination", 0.95),
                _metric("answer_relevancy", 0.95),
                _metric("contextual_precision", 0.4),
                _metric("contextual_recall", 0.95),
            ],
            pass_threshold=0.5,
        )
        decision = engine.evaluate(report)
        assert decision.status == QualityStatus.WARNING
        assert QualityRuleId.MIN_CONTEXT_PRECISION.value in decision.warnings

    def test_fail_on_tool_failures(self) -> None:
        engine = QualityGateEngine(settings=_settings(max_tool_failures=0))
        report = _report(tool_passed=False)
        decision = engine.evaluate(report, tool_failure_count=2)
        assert decision.status == QualityStatus.FAIL
        assert QualityRuleId.MAX_TOOL_FAILURES.value in decision.failed_rules

    def test_warning_on_high_latency(self) -> None:
        engine = QualityGateEngine(settings=_settings(max_total_latency=50.0))
        decision = engine.evaluate(_report(latency=500.0))
        assert decision.status == QualityStatus.WARNING
        assert QualityRuleId.MAX_TOTAL_LATENCY.value in decision.warnings

    def test_warning_on_high_cost(self) -> None:
        engine = QualityGateEngine(settings=_settings(max_cost=0.001))
        decision = engine.evaluate(_report(cost=0.5))
        assert decision.status == QualityStatus.WARNING
        assert QualityRuleId.MAX_COST.value in decision.warnings

    def test_disabled_raises_by_default(self) -> None:
        engine = QualityGateEngine(settings=_settings(quality_gate_enabled=False))
        with pytest.raises(QualityGateDisabledError):
            engine.evaluate(_report())

    def test_disabled_auto_pass_when_allowed(self) -> None:
        engine = QualityGateEngine(
            settings=_settings(quality_gate_enabled=False),
            allow_when_disabled=True,
        )
        decision = engine.evaluate(_report())
        assert decision.status == QualityStatus.PASS
        assert "disabled" in decision.reason.lower()

    def test_batch_fail_propagates(self) -> None:
        engine = QualityGateEngine(settings=_settings(min_faithfulness=0.95))
        good = _report(question="q1")
        bad = _report(
            question="q2",
            overall_via_metrics=[
                _metric("faithfulness", 0.2),
                _metric("hallucination", 0.95),
                _metric("answer_relevancy", 0.95),
                _metric("contextual_precision", 0.95),
                _metric("contextual_recall", 0.95),
            ],
        )
        decision = engine.evaluate_batch([good, bad])
        assert decision.status == QualityStatus.FAIL
        assert decision.metadata["fail_count"] >= 1


class TestQualityGateConfiguration:
    def test_thresholds_from_settings(self) -> None:
        thresholds = QualityGateThresholds.from_settings(
            _settings(min_faithfulness=0.88, max_hallucination=0.15)
        )
        assert thresholds.min_faithfulness == 0.88
        assert thresholds.max_hallucination == 0.15
        rules = build_default_rules(thresholds)
        assert any(r.rule_id == QualityRuleId.MIN_FAITHFULNESS for r in rules)

    def test_gate_skips_missing_metrics(self) -> None:
        gate = QualityGate(
            thresholds=QualityGateThresholds(skip_missing_metrics=True)
        )
        # Only overall score metrics present — faithfulness rule skipped.
        report = _report(
            overall_via_metrics=[_metric("answer_length", 0.9)]
        )
        results = gate.evaluate(report)
        faithfulness = next(
            r for r in results if r.rule_id == QualityRuleId.MIN_FAITHFULNESS
        )
        assert faithfulness.skipped is True
        assert faithfulness.passed is True


class TestQualityDecisionAndRecommendations:
    def test_decision_fields(self) -> None:
        decision = QualityDecision(
            status=QualityStatus.FAIL,
            reason="FAIL — broken rules: min_faithfulness",
            failed_rules=["min_faithfulness"],
            warnings=[],
            overall_score=0.4,
            correlation_id="abc",
            timestamp=datetime.now(timezone.utc),
        )
        assert decision.is_blocking_failure is True
        assert decision.passed is False

    def test_recommendations_catalog(self) -> None:
        gate = QualityGate()
        report = _report(
            overall_via_metrics=[
                _metric("faithfulness", 0.2),
                # Low higher-is-better score ⇒ high raw hallucination → fail max rule
                _metric("hallucination", 0.1),
                _metric("answer_relevancy", 0.9),
            ]
        )
        results = gate.evaluate(report)
        # Force fail severity outcomes into recommendations.
        for result in results:
            if result.rule_id in {
                QualityRuleId.MIN_FAITHFULNESS,
                QualityRuleId.MAX_HALLUCINATION,
            }:
                assert result.passed is False
        recs = recommendations_for_results(results)
        assert len(recs) >= 2
        categories = {r.category for r in recs}
        assert "retrieval" in categories or "prompt" in categories


class TestBenchmarkHistory:
    def test_append_compare_and_dashboard(self, tmp_path: Path) -> None:
        history = BenchmarkHistory(tmp_path / "history.json")
        summary1 = BenchmarkSummary(
            example_count=2,
            pass_rate=1.0,
            overall_average_score=0.9,
            average_latency_ms=100.0,
            average_cost_usd=0.01,
        )
        summary2 = BenchmarkSummary(
            example_count=2,
            pass_rate=0.5,
            overall_average_score=0.6,
            average_latency_ms=200.0,
            average_cost_usd=0.02,
        )
        r1 = BenchmarkRunRecord.from_summary(
            summary1,
            suite_name="demo",
            quality_status=QualityStatus.PASS,
            tool_validation_pass_rate=1.0,
        )
        r2 = BenchmarkRunRecord.from_summary(
            summary2,
            suite_name="demo",
            quality_status=QualityStatus.FAIL,
            tool_validation_pass_rate=0.5,
        )
        history.append(r1)
        history.append(r2)
        assert history.latest(suite_name="demo") is not None
        comparison = history.compare(r2, suite_name="demo")
        assert comparison is not None
        assert comparison.trend == "degrading"
        assert comparison.score_delta is not None and comparison.score_delta < 0

        dashboard = BenchmarkDashboardModel.from_history(history.load())
        assert dashboard.evaluation_count == 2
        assert dashboard.failure_count == 1
        assert dashboard.pass_count == 1
        assert len(dashboard.trend_data) == 2
        assert len(dashboard.historical_scores) == 2


class TestRegressionEngine:
    def test_detects_score_latency_tool_prompt_cost(self) -> None:
        runner = RegressionRunner(
            score_drop_threshold=0.05,
            latency_increase_ratio=0.2,
            cost_increase_ratio=0.2,
        )
        previous = [
            _report(
                question="q",
                latency=100.0,
                cost=0.01,
                answer="old answer",
                overall_via_metrics=[
                    _metric("faithfulness", 0.95),
                    _metric("hallucination", 0.95),
                    _metric("answer_relevancy", 0.95),
                ],
                tool_passed=True,
            )
        ]
        current = [
            _report(
                question="q",
                latency=200.0,
                cost=0.05,
                answer="new answer",
                overall_via_metrics=[
                    _metric("faithfulness", 0.5),
                    _metric("hallucination", 0.95),
                    _metric("answer_relevancy", 0.95),
                ],
                tool_passed=False,
            )
        ]
        report = runner.compare(previous, current)
        assert report.has_regressions is True
        assert report.summary["score_regression_count"] >= 1
        assert report.summary["latency_regression_count"] == 1
        assert report.summary["tool_regression_count"] == 1
        assert report.summary["prompt_regression_count"] == 1
        assert report.summary["cost_regression_count"] == 1
        assert isinstance(report.cost_regressions[0], CostRegression)


class TestContinuousEvaluation:
    def test_end_to_end_quality_pipeline(self, tmp_path: Path) -> None:
        settings = _settings(
            benchmark_history_path=str(tmp_path / "history.json"),
            quality_report_directory=str(tmp_path / "quality"),
        )
        service = ContinuousEvaluationService(settings=settings)
        result = service.evaluate(
            [_report()],
            suite_name="sprint12",
            correlation_id="corr-ce-1",
            write_reports=True,
            run_name="quality_report",
        )
        assert result.decision.status == QualityStatus.PASS
        assert result.benchmark_record is not None
        assert result.timeline.stage_names() == [
            "request",
            "evaluation",
            "quality_gate",
            "decision",
        ]
        assert (tmp_path / "quality" / "quality_report.json").is_file()
        assert (tmp_path / "quality" / "quality_report.html").is_file()
        assert (tmp_path / "quality" / "quality_report.csv").is_file()
        assert result.dashboard["evaluation_count"] == 1

    def test_with_previous_reports_regression(self, tmp_path: Path) -> None:
        settings = _settings(
            benchmark_history_path=str(tmp_path / "history.json"),
            quality_report_directory=str(tmp_path / "quality"),
            min_faithfulness=0.3,
        )
        service = ContinuousEvaluationService(settings=settings)
        previous = [
            _report(
                overall_via_metrics=[
                    _metric("faithfulness", 0.95),
                    _metric("hallucination", 0.95),
                    _metric("answer_relevancy", 0.95),
                ]
            )
        ]
        current = [
            _report(
                overall_via_metrics=[
                    _metric("faithfulness", 0.5),
                    _metric("hallucination", 0.95),
                    _metric("answer_relevancy", 0.95),
                ]
            )
        ]
        result = service.evaluate(
            current,
            previous_reports=previous,
            write_reports=False,
        )
        assert result.regression_report is not None
        assert result.regression_report.has_regressions is True


class TestArchitecturalImprovements:
    def test_planner_decision_round_trip(self) -> None:
        output = PlannerOutput(
            required_tools=["calculator"],
            execution_order=["calculator"],
            rationale="math",
        )
        decision = output.to_decision(confidence=0.9)
        assert isinstance(decision, PlannerDecision)
        assert decision.selected_tools == ["calculator"]
        assert decision.confidence == 0.9
        assert decision.execution_strategy == ExecutionStrategy.SEQUENTIAL
        back = decision.to_planner_output()
        assert back.selected_tools == ["calculator"]

    def test_tool_result_and_metrics(self) -> None:
        metrics = ToolExecutionMetrics.from_latency(
            execution_time_ms=12.5,
            status=ToolExecutionStatus.SUCCESS,
            queue_time_ms=1.0,
            retries=0,
        )
        typed = ToolResult.ok(
            "calculator",
            CalculatorResultData(result=42, expression="6*7"),
            metrics=metrics,
        )
        assert typed.success is True
        assert typed.data is not None
        assert typed.data.result == 42
        execution = ToolExecution(
            tool_name="calculator",
            arguments={"expression": "6*7"},
            output={"result": 42},
            latency_ms=12.5,
            status=ToolExecutionStatus.SUCCESS,
            metrics=metrics,
        )
        assert execution.ensure_metrics().execution_time_ms == 12.5

    def test_evaluation_timeline_stages(self) -> None:
        timeline = EvaluationTimeline()
        timeline.add(TimelineStage.REQUEST, detail="start")
        timeline.add(TimelineStage.PLANNING, detail="plan")
        timeline.add(TimelineStage.TOOL_EXECUTION, detail="tools")
        timeline.add(TimelineStage.LLM, detail="llm")
        timeline.add(TimelineStage.EVALUATION, detail="eval")
        timeline.add(TimelineStage.QUALITY_GATE, detail="gate")
        timeline.add(TimelineStage.DECISION, detail="PASS")
        assert timeline.stage_names() == [
            "request",
            "planning",
            "tool_execution",
            "llm",
            "evaluation",
            "quality_gate",
            "decision",
        ]


class TestQualityReportWriter:
    def test_writes_json_csv_html(self, tmp_path: Path) -> None:
        writer = QualityReportWriter(tmp_path)
        decision = QualityDecision(
            status=QualityStatus.PASS,
            reason="PASS — all rules satisfied",
            failed_rules=[],
            warnings=[],
            overall_score=0.9,
            correlation_id="r1",
        )
        paths = writer.write_all(
            decision=decision,
            reports=[_report()],
            benchmark_history={"trend": "stable"},
            regression_summary={"has_regressions": False},
            run_name="quality_report",
        )
        assert paths["json"].is_file()
        assert paths["csv"].is_file()
        assert paths["html"].is_file()
        html = paths["html"].read_text(encoding="utf-8")
        assert "PASS" in html
        assert "Recommendations" in html


class TestBenchmarkRunnerStillWorks:
    def test_summarize_unchanged(self) -> None:
        runner = BenchmarkRunner()
        summary = runner.summarize([_report(), _report(question="other?")])
        assert summary.example_count == 2
        assert summary.pass_rate == 1.0
        assert summary.average_faithfulness is not None
