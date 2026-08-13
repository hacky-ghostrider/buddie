"""Canonical demo runner — Agent → Trace → DeepEval → Gates → Reports.

Extracted from ``scripts/demo.py`` so FastAPI and the CLI share one path.
Does not add new AI capabilities; it composes existing services only.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app.agent.service import AgentService
from app.config.settings import Settings, get_settings
from app.demo.models import DemoResult
from app.evaluation.automation.costing import estimate_cost_usd
from app.evaluation.automation.report_writer import EvaluationReportWriter
from app.evaluation.benchmark.runner import BenchmarkRunner
from app.evaluation.continuous import ContinuousEvaluationService
from app.evaluation.dataset.loader import GoldenDatasetLoader
from app.evaluation.deepeval import DeepEvalMetricAdapter, DeepEvalMetricName
from app.evaluation.evaluator import EvaluationService
from app.evaluation.metrics import AnswerLengthMetric, ContextCountMetric
from app.evaluation.registry import MetricRegistry
from app.evaluation.scenarios import CANONICAL_DATASET_PATH, CANONICAL_SCENARIO_ID
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer
from app.tracing.service import TracingService

logger = logging.getLogger(__name__)

_OFFLINE_DEEPEVAL_SCORES: dict[str, float] = {
    DeepEvalMetricName.FAITHFULNESS.value: 0.92,
    DeepEvalMetricName.HALLUCINATION.value: 0.08,
    DeepEvalMetricName.ANSWER_RELEVANCY.value: 0.90,
    DeepEvalMetricName.CONTEXTUAL_PRECISION.value: 0.88,
    DeepEvalMetricName.CONTEXTUAL_RECALL.value: 0.85,
}


def _handbook_rag_response(question: str) -> RAGResponse:
    """Deterministic RAG response for the leave-policy demo question."""
    answer = (
        "The employee handbook leave policy defines paid time off eligibility, "
        "accrual, approval workflow, and notice requirements for vacation and "
        "related leave types."
    )
    return RAGResponse(
        question=question.strip(),
        answer=answer,
        retrieved_documents=[
            RetrievedDocument(
                id="handbook-leave-1",
                text=(
                    "Employees accrue paid leave per the handbook policy. "
                    "Vacation requires manager approval and advance notice."
                ),
                metadata={
                    "source": "employee_handbook.pdf",
                    "file_name": "employee_handbook.pdf",
                },
                score=0.93,
            )
        ],
        retrieval_metadata={"retrieved_count": 1, "demo": True},
        generation_metadata={
            "model": "demo-offline",
            "prompt": {"system": "demo", "user": question},
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
        },
        latency=LatencyBreakdown(
            retrieval_ms=12.0,
            prompt_build_ms=2.0,
            llm_ms=45.0,
            total_ms=59.0,
        ),
        correlation_id="demo-rag",
    )


def _build_offline_rag() -> MagicMock:
    """Mock RAGService used by agent tools (no OpenAI / Chroma)."""
    service = MagicMock()
    service.query.side_effect = lambda request: _handbook_rag_response(request.question)
    return service


def _offline_measure_fn(metric_name: str):
    """Return a DeepEval-compatible measure callable with fixed scores."""

    class _Measured:
        def __init__(self) -> None:
            self.score = _OFFLINE_DEEPEVAL_SCORES.get(metric_name, 0.85)
            self.reason = f"Offline demo score for {metric_name}"
            self.success = True
            self.threshold = 0.7

    def _measure(_test_case: Any) -> _Measured:
        return _Measured()

    return _measure


def _build_offline_registry(settings: Settings) -> MetricRegistry:
    """Registry with placeholders + DeepEval adapters using offline measure fns."""
    registry = MetricRegistry()
    registry.register(AnswerLengthMetric())
    registry.register(ContextCountMetric())
    for name in DeepEvalMetricName:
        registry.register(
            DeepEvalMetricAdapter(
                name,
                pass_threshold=settings.default_pass_threshold,
                measure_fn=_offline_measure_fn(str(name)),
            )
        )
    return registry


def _build_rag_response_from_agent(agent_result) -> RAGResponse:
    """Reconstruct a RAGResponse from agent EvaluationContext for scoring."""
    ctx = agent_result.evaluation_context
    docs = list(ctx.retrieved_documents) if ctx and ctx.retrieved_documents else []
    if not docs:
        docs = [
            RetrievedDocument(
                id="demo-fallback",
                text="Employees accrue paid leave per the handbook policy.",
                metadata={"source": "employee_handbook.pdf"},
                score=0.9,
            )
        ]
    tokens = (
        dict(ctx.token_usage)
        if ctx and ctx.token_usage
        else {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200,
        }
    )
    model = (ctx.model if ctx and ctx.model else None) or "agent-demo"
    return RAGResponse(
        question=agent_result.question,
        answer=agent_result.final_answer,
        retrieved_documents=docs,
        retrieval_metadata={"source": "agent"},
        generation_metadata={
            "model": model,
            "prompt": (ctx.prompt if ctx else None) or {},
            **tokens,
        },
        latency=LatencyBreakdown(
            retrieval_ms=0.0,
            prompt_build_ms=0.0,
            llm_ms=0.0,
            total_ms=agent_result.latency_ms or 0.0,
        ),
        correlation_id=agent_result.correlation_id,
    )


def _write_benchmark_csv(summary, path: Path) -> None:
    """Write a flat benchmark.csv for interview artifacts."""
    row = {
        "example_count": summary.example_count,
        "pass_rate": summary.pass_rate,
        "overall_average_score": summary.overall_average_score,
        "average_faithfulness": summary.average_faithfulness,
        "average_hallucination": summary.average_hallucination,
        "average_relevancy": summary.average_relevancy,
        "average_context_precision": summary.average_context_precision,
        "average_context_recall": summary.average_context_recall,
        "average_latency_ms": summary.average_latency_ms,
        "average_cost_usd": summary.average_cost_usd,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def run_canonical_demo(
    *,
    live: bool = False,
    dataset: str | Path | None = None,
    output_dir: str | Path = "./data/demo",
) -> DemoResult:
    """Execute the full interview demonstration pipeline.

    Args:
        live: Use real RAG / DeepEval / LangSmith when True.
        dataset: Path to the canonical golden dataset.
        output_dir: Directory for demo artifacts.

    Returns:
        Structured ``DemoResult`` for API / CLI consumers.
    """
    get_settings.cache_clear()
    base = get_settings()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = out / "reports"
    quality_dir = out / "quality"
    benchmark_dir = out / "benchmarks"
    report_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    settings = base.model_copy(
        update={
            "report_directory": str(report_dir),
            "quality_report_directory": str(quality_dir),
            "benchmark_directory": str(benchmark_dir),
            "benchmark_history_path": str(benchmark_dir / "history.json"),
            "enable_langsmith": live and base.enable_langsmith,
            "enable_deepeval": True,
            "enable_tool_validation": True,
            "quality_gate_enabled": True,
        }
    )

    dataset_path = str(dataset or CANONICAL_DATASET_PATH)
    example = GoldenDatasetLoader().load_one(
        dataset_path, example_id=CANONICAL_SCENARIO_ID
    )
    logger.info(
        "Demo loaded scenario=%s expected_tools=%s live=%s",
        CANONICAL_SCENARIO_ID,
        example.expected_tools,
        live,
    )

    if live:
        from app.api.deps import get_rag_service

        rag = get_rag_service()
        tracing = TracingService(settings=settings)
    else:
        rag = _build_offline_rag()
        tracing = TracingService(tracer=NoOpTracer(), settings=settings)

    agent = AgentService(
        rag_service=rag,
        tracing_service=tracing,
        settings=settings,
    )
    agent_result = agent.run(
        example.question,
        metadata={
            "scenario": CANONICAL_SCENARIO_ID,
            "golden_id": example.id,
            "demo": True,
            "mode": "live" if live else "offline",
        },
        expected_answer=example.expected_answer,
        expected_sources=example.expected_sources,
    )

    if live and settings.enable_deepeval:
        from app.evaluation.evaluator import create_default_registry

        registry = create_default_registry(settings)
    else:
        registry = _build_offline_registry(settings)

    evaluation_service = EvaluationService(registry=registry, settings=settings)
    rag_response = _build_rag_response_from_agent(agent_result)
    eval_report = evaluation_service.evaluate(
        example.question,
        rag_response,
        expected_answer=example.expected_answer,
        expected_sources=example.expected_sources,
        metadata={
            "golden_id": example.id,
            "scenario": CANONICAL_SCENARIO_ID,
            "correlation_id": agent_result.correlation_id,
        },
    )
    cost = estimate_cost_usd(
        eval_report.token_usage,
        input_cost_per_1k=settings.input_token_cost_per_1k,
        output_cost_per_1k=settings.output_token_cost_per_1k,
    )
    report = eval_report.model_copy(
        update={
            "tool_validation": agent_result.tool_validation,
            "langsmith_run_id": agent_result.run_id,
            "langsmith_trace_id": agent_result.trace_id,
            "langsmith_run_url": agent_result.run_url,
            "estimated_cost_usd": cost,
            "rag_latency_ms": agent_result.latency_ms,
            "passed": eval_report.passed
            and (
                agent_result.tool_validation is None
                or agent_result.tool_validation.passed
            ),
            "metadata": {
                **eval_report.metadata,
                "correlation_id": agent_result.correlation_id,
                "scenario": CANONICAL_SCENARIO_ID,
            },
        }
    )

    continuous = ContinuousEvaluationService(settings=settings)
    continuous_result = continuous.evaluate(
        [report],
        suite_name="demo",
        correlation_id=agent_result.correlation_id,
        write_reports=True,
        run_name="demo_quality",
        metadata={"scenario": CANONICAL_SCENARIO_ID},
    )

    writer = EvaluationReportWriter(report_dir)
    eval_paths = writer.write_all([report], run_name="evaluation")
    bench = BenchmarkRunner()
    summary = bench.summarize(
        [report],
        metadata={"scenario": CANONICAL_SCENARIO_ID, "suite": "demo"},
    )
    bench_csv = benchmark_dir / "benchmark.csv"
    bench_json = bench.write_summary(summary, benchmark_dir, run_name="benchmark")
    _write_benchmark_csv(summary, bench_csv)

    output_paths: dict[str, str] = {
        **{f"evaluation_{k}": str(v) for k, v in eval_paths.items()},
        **{f"quality_{k}": str(v) for k, v in continuous_result.output_paths.items()},
        "benchmark_json": str(bench_json),
        "benchmark_csv": str(bench_csv),
    }

    return DemoResult(
        scenario_id=CANONICAL_SCENARIO_ID,
        mode="live" if live else "offline",
        question=example.question,
        expected_tools=list(example.expected_tools),
        agent_result=agent_result,
        evaluation_report=report,
        quality_decision=continuous_result.decision,
        report_paths=output_paths,
        metadata={
            "golden_id": example.id,
            "correlation_id": agent_result.correlation_id,
        },
    )


__all__ = ["run_canonical_demo"]
