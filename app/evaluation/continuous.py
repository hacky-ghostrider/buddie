"""Continuous evaluation service — Agent → Eval → Gates → History → Reports.

Sprint 12 orchestration layer that composes existing Sprint 9–11 services
without refactoring them:

    Agent / RAG evaluation
            ↓
    DeepEval + Tool Validation + LangSmith (via EvaluationReport)
            ↓
    Regression comparison (optional previous run)
            ↓
    QualityGateEngine → PASS / WARNING / FAIL
            ↓
    BenchmarkHistory append + QualityReportWriter
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.evaluation.benchmark.dashboard import BenchmarkDashboardModel
from app.evaluation.benchmark.history import (
    BenchmarkHistory,
    BenchmarkRunRecord,
)
from app.evaluation.benchmark.runner import BenchmarkRunner
from app.evaluation.quality.decision import QualityDecision
from app.evaluation.quality.engine import QualityGateEngine
from app.evaluation.quality.report import QualityReportWriter
from app.evaluation.regression.runner import RegressionReport, RegressionRunner
from app.evaluation.report import EvaluationReport
from app.evaluation.timeline import EvaluationTimeline, TimelineStage
from app.evaluation.tool_validation.tool_execution import ToolExecutionStatus

logger = logging.getLogger(__name__)


@dataclass
class ContinuousEvaluationResult:
    """Outcome of one continuous-evaluation run."""

    decision: QualityDecision
    reports: list[EvaluationReport] = field(default_factory=list)
    benchmark_record: BenchmarkRunRecord | None = None
    regression_report: RegressionReport | None = None
    output_paths: dict[str, Path] = field(default_factory=dict)
    timeline: EvaluationTimeline = field(default_factory=EvaluationTimeline)
    dashboard: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""


class ContinuousEvaluationService:
    """Apply quality gates, regression, and benchmark history to reports.

    This service does **not** replace ``EvaluationAutomationService`` or
    ``AgentService``. Callers produce ``EvaluationReport``s first, then
    hand them here for continuous quality decisions.

    Args:
        settings: Application settings.
        gate_engine: Optional quality-gate engine.
        regression_runner: Optional regression comparator.
        benchmark_runner: Optional suite aggregator.
        history: Optional benchmark history store.
        report_writer: Optional quality report writer.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        gate_engine: QualityGateEngine | None = None,
        regression_runner: RegressionRunner | None = None,
        benchmark_runner: BenchmarkRunner | None = None,
        history: BenchmarkHistory | None = None,
        report_writer: QualityReportWriter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._gate_engine = gate_engine or QualityGateEngine(
            settings=self._settings,
            allow_when_disabled=True,
        )
        self._regression_runner = regression_runner or RegressionRunner()
        self._benchmark_runner = benchmark_runner or BenchmarkRunner()
        self._history = history or BenchmarkHistory(
            self._settings.benchmark_history_path
        )
        self._report_writer = report_writer or QualityReportWriter(
            self._settings.quality_report_directory
        )

    def evaluate(
        self,
        reports: list[EvaluationReport],
        *,
        previous_reports: list[EvaluationReport] | None = None,
        suite_name: str = "default",
        correlation_id: str | None = None,
        write_reports: bool = True,
        run_name: str = "quality_report",
        metadata: dict[str, Any] | None = None,
    ) -> ContinuousEvaluationResult:
        """Run continuous evaluation over one or more reports.

        Workflow:
            Request → Evaluation (already done) → Quality Gate → Decision
            → Benchmark History → Reports

        Args:
            reports: Current evaluation reports.
            previous_reports: Optional previous run for regression.
            suite_name: Logical suite name for history.
            correlation_id: Optional correlation id.
            write_reports: Persist quality_report.* artifacts.
            run_name: Output stem.
            metadata: Optional extras.

        Returns:
            ``ContinuousEvaluationResult`` with decision and artifacts.
        """
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        corr = correlation_id or run_id
        timeline = EvaluationTimeline()
        timeline.add(
            TimelineStage.REQUEST,
            detail=f"Continuous evaluation started suite={suite_name}",
            metadata={"report_count": len(reports)},
        )
        timeline.add(
            TimelineStage.EVALUATION,
            detail="Evaluation reports received",
            metadata={"questions": [r.question for r in reports]},
        )

        # Derive tool signals from reports for gate rules.
        tool_failures = self._count_tool_failures(reports)
        max_tool_latency = self._max_tool_latency(reports)

        if len(reports) == 1:
            decision = self._gate_engine.evaluate(
                reports[0],
                correlation_id=corr,
                tool_failure_count=tool_failures,
                max_tool_latency_ms=max_tool_latency,
                metadata=metadata,
            )
        else:
            decision = self._gate_engine.evaluate_batch(
                reports,
                correlation_id=corr,
                metadata=metadata,
            )

        timeline.add(
            TimelineStage.QUALITY_GATE,
            detail=f"Quality gate → {decision.status.value}",
            metadata={
                "failed_rules": decision.failed_rules,
                "warnings": decision.warnings,
            },
        )
        timeline.add(
            TimelineStage.DECISION,
            detail=decision.reason,
            metadata={"status": decision.status.value},
        )

        regression_report = None
        if previous_reports:
            regression_report = self._regression_runner.compare(
                previous_reports,
                reports,
                previous_path="previous",
                current_path="current",
            )
            logger.info(
                "Regression comparison: has_regressions=%s summary=%s",
                regression_report.has_regressions,
                regression_report.summary,
            )

        summary = self._benchmark_runner.summarize(
            reports,
            metadata={
                "suite_name": suite_name,
                "quality_status": decision.status.value,
                "run_id": run_id,
                **(metadata or {}),
            },
        )
        tool_pass_rate = self._tool_validation_pass_rate(reports)
        record = BenchmarkRunRecord.from_summary(
            summary,
            suite_name=suite_name,
            run_id=run_id,
            quality_status=decision.status,
            tool_validation_pass_rate=tool_pass_rate,
            correlation_id=corr,
            metadata=dict(metadata or {}),
        )
        self._history.append(record)
        comparison = self._history.compare(record, suite_name=suite_name)
        dashboard = BenchmarkDashboardModel.from_history(
            self._history.load()
        ).model_dump(mode="json")

        regression_summary: dict[str, Any] = {}
        if regression_report is not None:
            regression_summary = {
                "has_regressions": regression_report.has_regressions,
                "summary": regression_report.summary,
                "matched_questions": regression_report.matched_questions,
            }
        if comparison is not None:
            regression_summary["benchmark_trend"] = comparison.model_dump(
                mode="json"
            )

        paths: dict[str, Path] = {}
        if write_reports:
            paths = self._report_writer.write_all(
                decision=decision,
                reports=reports,
                benchmark_history={
                    "latest": record.model_dump(mode="json"),
                    "comparison": (
                        None
                        if comparison is None
                        else comparison.model_dump(mode="json")
                    ),
                    "dashboard": dashboard,
                },
                regression_summary=regression_summary,
                run_name=run_name,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Continuous evaluation completed: run_id=%s status=%s "
            "reports=%d elapsed_ms=%.1f",
            run_id,
            decision.status.value,
            len(reports),
            elapsed_ms,
        )
        return ContinuousEvaluationResult(
            decision=decision,
            reports=reports,
            benchmark_record=record,
            regression_report=regression_report,
            output_paths=paths,
            timeline=timeline,
            dashboard=dashboard,
            run_id=run_id,
        )

    @staticmethod
    def _count_tool_failures(reports: list[EvaluationReport]) -> int:
        """Count tool-validation / execution failures across reports."""
        failures = 0
        for report in reports:
            if report.tool_validation is not None and not report.tool_validation.passed:
                failures += len(report.tool_validation.failures or []) or 1
            # Also inspect embedded ToolExecution statuses in metadata context.
            ctx = (report.metadata or {}).get("evaluation_context") or {}
            for call in ctx.get("tool_calls") or []:
                if isinstance(call, dict):
                    status = str(call.get("status") or "").lower()
                    if status and status != ToolExecutionStatus.SUCCESS.value:
                        if status in {
                            ToolExecutionStatus.FAILED.value,
                            ToolExecutionStatus.TIMEOUT.value,
                        }:
                            failures += 1
        return failures

    @staticmethod
    def _max_tool_latency(reports: list[EvaluationReport]) -> float | None:
        """Return the max observed tool latency from report metadata."""
        latencies: list[float] = []
        for report in reports:
            meta = report.metadata or {}
            raw = meta.get("max_tool_latency_ms")
            if raw is not None:
                try:
                    latencies.append(float(raw))
                except (TypeError, ValueError):
                    pass
            ctx = meta.get("evaluation_context") or {}
            for call in ctx.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("latency_ms") is not None:
                    try:
                        latencies.append(float(call["latency_ms"]))
                    except (TypeError, ValueError):
                        pass
        return max(latencies) if latencies else None

    @staticmethod
    def _tool_validation_pass_rate(reports: list[EvaluationReport]) -> float | None:
        """Compute tool-validation pass rate when validation is present."""
        scored = [r for r in reports if r.tool_validation is not None]
        if not scored:
            return None
        passed = sum(1 for r in scored if r.tool_validation and r.tool_validation.passed)
        return round(passed / len(scored), 6)


__all__ = ["ContinuousEvaluationResult", "ContinuousEvaluationService"]
