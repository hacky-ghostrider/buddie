"""Evaluation automation pipeline — RAG → Trace → DeepEval → Tools → Report.

WHY
---
Sprint 9 gave us pluggable metrics. Sprint 10 wires the *automation*:
load goldens, run RAG, emit LangSmith traces, score with DeepEval adapters,
validate tools, and write multi-format reports. This is the CI/offline
harness an AI Evaluation Engineer owns day-to-day.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.config.settings import Settings, get_settings
from app.evaluation.automation.costing import estimate_cost_usd
from app.evaluation.automation.report_writer import EvaluationReportWriter
from app.evaluation.context import EvaluationContext
from app.evaluation.dataset.loader import GoldenDatasetLoader
from app.evaluation.evaluator import EvaluationService, create_default_registry
from app.evaluation.models import GoldenExample
from app.evaluation.report import EvaluationReport
from app.evaluation.tool_validation.models import ActualToolCall
from app.evaluation.tool_validation.tool_contract import contracts_from_golden_fields
from app.evaluation.tool_validation.tool_execution import ToolExecution
from app.evaluation.tool_validation.trace_mapper import ToolTraceMapper
from app.evaluation.tool_validation.validator import ToolValidator
from app.orchestration.models import RAGRequest, RAGResponse
from app.tracing.service import TracingService, create_tracer

logger = logging.getLogger(__name__)


class RAGRunner(Protocol):
    """Minimal protocol for running RAG (real ``RAGService`` or test double)."""

    def query(self, request: RAGRequest) -> RAGResponse:
        """Execute one RAG query."""


@dataclass
class AutomationRunResult:
    """Outcome of a single-question or batch evaluation run."""

    reports: list[EvaluationReport] = field(default_factory=list)
    output_paths: dict[str, Path] = field(default_factory=dict)
    run_name: str = ""


class EvaluationAutomationService:
    """Orchestrate offline evaluation automation with DI collaborators.

    Workflow per golden example:
        Load example → Run RAG → DeepEval (via EvaluationService) →
        Tool Validator → LangSmith Trace → Enrich report → Persist.

    Args:
        rag_runner: Object with ``query(RAGRequest) → RAGResponse``.
        evaluation_service: Metric evaluation service.
        tracing_service: Trace recorder façade.
        tool_validator: Tool-call validator.
        dataset_loader: Golden dataset loader.
        report_writer: Multi-format report writer.
        settings: Application settings.
    """

    def __init__(
        self,
        *,
        rag_runner: RAGRunner,
        evaluation_service: EvaluationService | None = None,
        tracing_service: TracingService | None = None,
        tool_validator: ToolValidator | None = None,
        dataset_loader: GoldenDatasetLoader | None = None,
        report_writer: EvaluationReportWriter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._rag_runner = rag_runner
        registry = create_default_registry(self._settings)
        self._evaluation_service = evaluation_service or EvaluationService(
            registry=registry,
            settings=self._settings,
        )
        self._tracing_service = tracing_service or TracingService(
            tracer=create_tracer(self._settings),
            settings=self._settings,
        )
        self._tool_validator = tool_validator or ToolValidator()
        self._dataset_loader = dataset_loader or GoldenDatasetLoader()
        self._report_writer = report_writer or EvaluationReportWriter(
            self._settings.report_directory
        )

    def run_single(
        self,
        example: GoldenExample,
        *,
        actual_tool_calls: list[ActualToolCall] | None = None,
        tool_executions: list[ToolExecution] | None = None,
        langsmith_trace: Any | None = None,
        write_reports: bool = True,
        run_name: str | None = None,
    ) -> AutomationRunResult:
        """Evaluate one golden example end-to-end.

        Args:
            example: Golden dataset example.
            actual_tool_calls: Optional observed tool calls (agents later).
            tool_executions: Optional normalized executions (preferred).
            langsmith_trace: Optional LangSmith-like payload mapped via
                ``ToolTraceMapper`` (never parsed by ``ToolValidator``).
            write_reports: Whether to persist JSON/CSV/HTML.
            run_name: Optional output stem.

        Returns:
            ``AutomationRunResult`` with one report.
        """
        report = self._evaluate_example(
            example,
            actual_tool_calls=actual_tool_calls,
            tool_executions=tool_executions,
            langsmith_trace=langsmith_trace,
        )
        paths: dict[str, Path] = {}
        name = run_name or "evaluation_single"
        if write_reports:
            paths = self._report_writer.write_all([report], run_name=name)
        return AutomationRunResult(reports=[report], output_paths=paths, run_name=name)

    def run_batch(
        self,
        examples: list[GoldenExample],
        *,
        actual_tool_calls_by_id: dict[str, list[ActualToolCall]] | None = None,
        write_reports: bool = True,
        run_name: str | None = None,
    ) -> AutomationRunResult:
        """Evaluate a batch of golden examples.

        Args:
            examples: Golden examples.
            actual_tool_calls_by_id: Optional map of example id → tool calls.
            write_reports: Whether to persist artifacts.
            run_name: Optional output stem.

        Returns:
            ``AutomationRunResult`` with all reports.
        """
        reports: list[EvaluationReport] = []
        call_map = actual_tool_calls_by_id or {}
        for example in examples:
            key = example.id or example.question
            reports.append(
                self._evaluate_example(
                    example,
                    actual_tool_calls=call_map.get(key),
                )
            )
        paths: dict[str, Path] = {}
        name = run_name or "evaluation_batch"
        if write_reports:
            paths = self._report_writer.write_all(reports, run_name=name)
        logger.info(
            "Batch evaluation completed: examples=%d passed=%d",
            len(reports),
            sum(1 for r in reports if r.passed),
        )
        return AutomationRunResult(reports=reports, output_paths=paths, run_name=name)

    def run_from_dataset(
        self,
        dataset_path: str | Path | None = None,
        *,
        question: str | None = None,
        example_id: str | None = None,
        index: int | None = None,
        write_reports: bool = True,
        run_name: str | None = None,
    ) -> AutomationRunResult:
        """Load dataset and run single or batch mode.

        Single mode when ``question``, ``example_id``, or ``index`` is set;
        otherwise batch mode over the full dataset.
        """
        path = dataset_path or self._settings.golden_dataset_path
        if question is not None or example_id is not None or index is not None:
            example = self._dataset_loader.load_one(
                path,
                question=question,
                example_id=example_id,
                index=index,
            )
            return self.run_single(
                example,
                write_reports=write_reports,
                run_name=run_name or "evaluation_single",
            )
        examples = self._dataset_loader.load(path)
        return self.run_batch(
            examples,
            write_reports=write_reports,
            run_name=run_name or "evaluation_batch",
        )

    def _evaluate_example(
        self,
        example: GoldenExample,
        *,
        actual_tool_calls: list[ActualToolCall] | None = None,
        tool_executions: list[ToolExecution] | None = None,
        langsmith_trace: Any | None = None,
    ) -> EvaluationReport:
        """Run the full per-example automation workflow."""
        started = time.perf_counter()
        rag_response = self._rag_runner.query(
            RAGRequest(question=example.question)
        )

        # Adapter boundary: LangSmith / future agent traces → ToolExecution
        # → ActualToolCall. ToolValidator never parses LangSmith objects.
        mapper = ToolTraceMapper()
        executions = list(tool_executions or [])
        if not executions and langsmith_trace is not None:
            executions = mapper.map_langsmith_trace(langsmith_trace)
        resolved_calls = list(actual_tool_calls or [])
        if not resolved_calls and executions:
            resolved_calls = mapper.to_actual_tool_calls(executions)

        eval_report = self._evaluation_service.evaluate(
            example.question,
            rag_response,
            expected_answer=example.expected_answer,
            expected_sources=example.expected_sources,
            metadata={
                "golden_id": example.id,
                "category": example.category,
                "difficulty": example.difficulty,
                "tags": list(example.tags),
            },
        )

        tool_report = None
        contracts = contracts_from_golden_fields(
            expected_tools=example.expected_tools,
            expected_tool_arguments=example.expected_tool_arguments,
            expected_tool_order=example.expected_tool_order,
        )
        if self._settings.enable_tool_validation:
            if contracts and executions:
                tool_report = self._tool_validator.validate_contracts(
                    contracts,
                    executions,
                    metadata={"golden_id": example.id},
                )
            else:
                tool_report = self._tool_validator.validate_from_golden(
                    expected_tools=example.expected_tools,
                    expected_tool_arguments=example.expected_tool_arguments,
                    expected_tool_order=example.expected_tool_order,
                    actual_calls=resolved_calls,
                    metadata={"golden_id": example.id},
                )

        trace = self._tracing_service.trace_rag_evaluation(
            rag_response=rag_response,
            evaluation_report=eval_report,
            tool_validation=(
                tool_report.to_summary_dict() if tool_report is not None else None
            ),
            extra_metadata={
                "golden_id": example.id,
                "category": example.category,
                "scenario": example.id,
            },
        )

        gen_meta = dict(rag_response.generation_metadata or {})
        token_usage = {
            key: gen_meta[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in gen_meta
        } or dict(eval_report.token_usage)
        cost = estimate_cost_usd(
            token_usage,
            input_cost_per_1k=self._settings.input_token_cost_per_1k,
            output_cost_per_1k=self._settings.output_token_cost_per_1k,
        )

        # Single evaluation aggregate for future consumers (Sprint 11+).
        evaluation_context = EvaluationContext.from_rag_response(
            question=example.question,
            rag_response=rag_response,
            expected_answer=example.expected_answer,
            expected_sources=example.expected_sources,
            tool_calls=executions,
            tool_results=[ex.output for ex in executions],
            cost_usd=cost,
            langsmith_trace_id=trace.trace_id,
            langsmith_run_id=trace.run_id,
            langsmith_run_url=trace.run_url,
            metadata={
                "golden_id": example.id,
                "category": example.category,
                "difficulty": example.difficulty,
                "tags": list(example.tags),
                "expected_tools": list(example.expected_tools),
                "tool_contracts": [c.model_dump() for c in contracts],
            },
        )

        total_ms = (time.perf_counter() - started) * 1000.0
        enriched = EvaluationReport.build(
            question=eval_report.question,
            answer=eval_report.answer,
            retrieved_documents=list(eval_report.retrieved_documents),
            metrics=list(eval_report.metrics),
            latency_ms=total_ms,
            pass_threshold=eval_report.pass_threshold,
            expected_answer=example.expected_answer,
            rag_latency_ms=rag_response.latency.total_ms,
            token_usage=token_usage,
            estimated_cost_usd=cost,
            langsmith_run_id=trace.run_id,
            langsmith_trace_id=trace.trace_id,
            langsmith_run_url=trace.run_url,
            tool_validation=tool_report,
            metadata={
                **eval_report.metadata,
                "trace_enabled": trace.enabled,
                "automation_latency_ms": total_ms,
                "evaluation_context": evaluation_context.model_dump(
                    mode="json",
                    exclude={"retrieved_documents"},
                ),
            },
            evaluation_time=eval_report.evaluation_time,
        )
        logger.info(
            "Example evaluated: question=%r passed=%s score=%.4f run_url=%s",
            example.question[:80],
            enriched.passed,
            enriched.overall_score,
            enriched.langsmith_run_url,
        )
        return enriched


__all__ = [
    "RAGRunner",
    "AutomationRunResult",
    "EvaluationAutomationService",
]
