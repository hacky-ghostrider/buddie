"""Sprint 10 — AI Evaluation Automation tests (all vendors mocked).

No live OpenAI, DeepEval, or LangSmith calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.config.settings import Settings
from app.evaluation.automation import EvaluationAutomationService
from app.evaluation.automation.report_writer import EvaluationReportWriter, report_to_row
from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.dataset import GoldenDatasetLoader
from app.evaluation.deepeval import DeepEvalMetricAdapter, create_deepeval_metrics
from app.evaluation.deepeval.mapping import normalize_deepeval_score, to_metric_result
from app.evaluation.deepeval.metrics import DeepEvalMetricName
from app.evaluation.evaluator import EvaluationService, create_default_registry
from app.evaluation.models import GoldenExample, MetricResult
from app.evaluation.regression import RegressionRunner
from app.evaluation.report import EvaluationReport
from app.evaluation.tool_validation import ToolValidator
from app.evaluation.tool_validation.models import ActualToolCall, ToolCallExpectation
from app.orchestration.models import LatencyBreakdown, RAGRequest, RAGResponse
from app.retrieval.models import RetrievedDocument
from app.tracing.base import NoOpTracer, TraceSpanData
from app.tracing.langsmith_adapter import LangSmithTracer
from app.tracing.service import TracingService


def _settings(**overrides: Any) -> Settings:
    base = dict(
        app_env="test",
        enable_evaluation=True,
        enable_deepeval=False,
        enable_langsmith=False,
        enable_tool_validation=True,
        default_pass_threshold=0.7,
        metric_timeout=5.0,
        report_directory="./data/reports-test",
        benchmark_directory="./data/benchmarks-test",
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


def _doc(text: str = "Chunk about recursive splitting.", doc_id: str = "c1") -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        text=text,
        metadata={"file_name": "guide.pdf"},
        score=0.9,
    )


def _rag(
    question: str = "What is recursive character chunking?",
    answer: str = (
        "Recursive character chunking splits text using a hierarchy of "
        "separators while preserving overlap between adjacent chunks."
    ),
) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_documents=[_doc()],
        retrieval_metadata={},
        generation_metadata={
            "model": "gpt-4o-mini",
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "total_tokens": 160,
            "prompt": {"system": "sys", "user": question},
        },
        latency=LatencyBreakdown(
            retrieval_ms=8.0,
            prompt_build_ms=2.0,
            llm_ms=40.0,
            total_ms=50.0,
        ),
        correlation_id="corr-s10",
    )


class _FakeRAG:
    def __init__(self, response: RAGResponse | None = None) -> None:
        self.response = response or _rag()
        self.calls: list[RAGRequest] = []

    def query(self, request: RAGRequest) -> RAGResponse:
        self.calls.append(request)
        return self.response.model_copy(update={"question": request.question.strip()})


def _measured(score: float, *, success: bool = True, reason: str = "ok") -> MagicMock:
    metric = MagicMock()
    metric.score = score
    metric.success = success
    metric.reason = reason
    metric.threshold = 0.7
    return metric


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def test_golden_dataset_loader_reads_extended_schema(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "id": "ex-1",
                        "question": "Q1?",
                        "expected_answer": "A1",
                        "expected_sources": ["s1"],
                        "expected_tools": ["search_docs"],
                        "expected_tool_arguments": [{"query": "q"}],
                        "expected_tool_order": ["search_docs"],
                        "difficulty": "hard",
                        "category": "agent_foundation",
                        "tags": ["tools"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    examples = GoldenDatasetLoader().load(path)
    assert len(examples) == 1
    assert examples[0].expected_tools == ["search_docs"]
    assert examples[0].expected_tool_order == ["search_docs"]
    assert examples[0].category == "agent_foundation"


def test_golden_dataset_loader_rejects_missing_file(tmp_path: Path) -> None:
    from app.evaluation.exceptions import InvalidEvaluationInputError

    with pytest.raises(InvalidEvaluationInputError):
        GoldenDatasetLoader().load(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# DeepEval adapter (mocked measure)
# ---------------------------------------------------------------------------


def test_deepeval_adapter_normalizes_hallucination_inversion() -> None:
    assert normalize_deepeval_score(DeepEvalMetricName.HALLUCINATION, 0.2) == pytest.approx(
        0.8
    )
    assert normalize_deepeval_score(DeepEvalMetricName.FAITHFULNESS, 0.9) == pytest.approx(
        0.9
    )


def test_deepeval_adapter_evaluate_with_injected_measure() -> None:
    adapter = DeepEvalMetricAdapter(
        DeepEvalMetricName.FAITHFULNESS,
        measure_fn=lambda _tc: _measured(0.91),
    )
    from app.evaluation.models import EvaluationContext

    result = adapter.evaluate(
        EvaluationContext(
            question="What is chunking?",
            answer="Chunking splits text.",
            retrieved_documents=[_doc()],
        )
    )
    assert result.name == "faithfulness"
    assert result.score == pytest.approx(0.91)
    assert result.passed is True
    assert result.details["provider"] == "deepeval"


def test_create_deepeval_metrics_suite() -> None:
    metrics = create_deepeval_metrics(
        measure_fns={
            "faithfulness": lambda _tc: _measured(0.8),
            "hallucination": lambda _tc: _measured(0.1),
            "answer_relevancy": lambda _tc: _measured(0.85),
            "contextual_precision": lambda _tc: _measured(0.7),
            "contextual_recall": lambda _tc: _measured(0.75),
        }
    )
    assert [m.name() for m in metrics] == [
        "faithfulness",
        "hallucination",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
    ]


def test_contextual_metric_requires_expected_answer() -> None:
    adapter = DeepEvalMetricAdapter(
        DeepEvalMetricName.CONTEXTUAL_RECALL,
        measure_fn=lambda _tc: _measured(0.9),
    )
    from app.evaluation.models import EvaluationContext

    result = adapter.evaluate(
        EvaluationContext(
            question="Q?",
            answer="A",
            retrieved_documents=[_doc()],
        )
    )
    assert result.passed is False
    assert result.error is not None


def test_to_metric_result_maps_errors() -> None:
    result = to_metric_result(
        metric_name="faithfulness",
        raw_score=None,
        passed=False,
        pass_threshold=0.7,
        error="boom",
    )
    assert result.score == 0.0
    assert result.error == "boom"


# ---------------------------------------------------------------------------
# LangSmith adapter (mocked client)
# ---------------------------------------------------------------------------


def test_langsmith_tracer_records_run_with_mock_client() -> None:
    client = MagicMock()
    tracer = LangSmithTracer(
        project_name="test-project",
        client=client,
    )
    record = tracer.record(
        TraceSpanData(
            question="Q?",
            retrieved_chunks=["chunk"],
            prompt={"user": "Q?"},
            model="gpt-4o-mini",
            tokens={"total_tokens": 10},
            latency_ms=12.0,
            answer="A",
            evaluation_results={"overall_score": 0.9},
        )
    )
    assert record.enabled is True
    assert record.run_id is not None
    assert record.run_url is not None
    assert "test-project" == record.project
    client.create_run.assert_called_once()
    client.update_run.assert_called_once()


def test_noop_tracer_disabled() -> None:
    record = NoOpTracer().record(TraceSpanData(question="Q?"))
    assert record.enabled is False
    assert record.run_id is not None


def test_tracing_service_attaches_evaluation_results() -> None:
    client = MagicMock()
    service = TracingService(
        tracer=LangSmithTracer(project_name="p", client=client),
        settings=_settings(enable_langsmith=True),
    )
    report = EvaluationReport.build(
        question="Q?",
        answer="A",
        retrieved_documents=[_doc()],
        metrics=[
            MetricResult(name="faithfulness", score=0.9, passed=True),
        ],
        latency_ms=5.0,
        pass_threshold=0.7,
    )
    record = service.trace_rag_evaluation(
        rag_response=_rag(),
        evaluation_report=report,
    )
    assert record.enabled is True
    assert client.create_run.called


# ---------------------------------------------------------------------------
# Tool validator
# ---------------------------------------------------------------------------


def test_tool_validator_pass_order_and_args() -> None:
    validator = ToolValidator(allow_extra_calls=False)
    report = validator.validate(
        [
            ToolCallExpectation(
                tool_name="search_docs",
                arguments={"query": "vacation"},
                order=0,
            ),
            ToolCallExpectation(tool_name="summarize", order=1),
        ],
        [
            ActualToolCall(
                tool_name="search_docs",
                arguments={"query": "vacation", "extra": True},
                order=0,
                latency_ms=12.0,
            ),
            ActualToolCall(tool_name="summarize", arguments={}, order=1),
        ],
    )
    assert report.passed is True
    assert report.expected_tools == ["search_docs", "summarize"]


def test_tool_validator_fails_wrong_tool_and_order() -> None:
    validator = ToolValidator()
    report = validator.validate_from_golden(
        expected_tool_order=["search_docs", "summarize"],
        expected_tool_arguments=[{"query": "x"}, {}],
        actual_calls=[
            ActualToolCall(tool_name="summarize", order=0),
            ActualToolCall(
                tool_name="search_docs",
                arguments={"query": "x"},
                order=1,
            ),
        ],
    )
    assert report.passed is False
    assert report.failures


def test_tool_validator_vacuous_pass_without_expectations() -> None:
    report = ToolValidator().validate([], [])
    assert report.passed is True


# ---------------------------------------------------------------------------
# Automation + reports
# ---------------------------------------------------------------------------


def test_evaluation_automation_single_with_mocks(tmp_path: Path) -> None:
    settings = _settings(
        report_directory=str(tmp_path / "reports"),
        enable_deepeval=True,
        enable_langsmith=True,
        enable_tool_validation=True,
    )
    measure_fns = {
        name: (lambda _tc, s=score: _measured(s))
        for name, score in {
            "faithfulness": 0.9,
            "hallucination": 0.05,
            "answer_relevancy": 0.88,
            "contextual_precision": 0.8,
            "contextual_recall": 0.82,
        }.items()
    }
    registry = create_default_registry(
        _settings(enable_deepeval=False)
    )
    for adapter in create_deepeval_metrics(measure_fns=measure_fns):
        registry.register(adapter)

    client = MagicMock()
    tracing = TracingService(
        tracer=LangSmithTracer(project_name="eval", client=client),
        settings=settings,
    )
    service = EvaluationAutomationService(
        rag_runner=_FakeRAG(),
        evaluation_service=EvaluationService(registry=registry, settings=settings),
        tracing_service=tracing,
        tool_validator=ToolValidator(),
        report_writer=EvaluationReportWriter(tmp_path / "reports"),
        settings=settings,
    )
    example = GoldenExample(
        id="rag-1",
        question="What is recursive character chunking?",
        expected_answer=(
            "Recursive character chunking splits text using a hierarchy of "
            "separators while preserving overlap between adjacent chunks."
        ),
        expected_sources=["guide.pdf"],
        category="rag",
    )
    result = service.run_single(example, run_name="unit_single")
    assert len(result.reports) == 1
    report = result.reports[0]
    assert report.langsmith_run_url is not None
    assert report.expected_answer is not None
    assert "json" in result.output_paths
    assert result.output_paths["json"].is_file()
    assert result.output_paths["csv"].is_file()
    assert result.output_paths["html"].is_file()
    row = report_to_row(report)
    assert "deepeval_metrics" in row
    assert "langsmith_trace_url" in row


def test_report_writer_includes_required_fields(tmp_path: Path) -> None:
    report = EvaluationReport.build(
        question="Q?",
        expected_answer="E",
        answer="A",
        retrieved_documents=[_doc()],
        metrics=[MetricResult(name="faithfulness", score=0.9, passed=True)],
        latency_ms=11.0,
        pass_threshold=0.7,
        rag_latency_ms=50.0,
        token_usage={"total_tokens": 160},
        langsmith_run_url="https://smith.langchain.com/runs/abc",
        langsmith_run_id="abc",
        langsmith_trace_id="abc",
    )
    writer = EvaluationReportWriter(tmp_path)
    paths = writer.write_all([report], run_name="fields")
    html = paths["html"].read_text(encoding="utf-8")
    assert "LangSmith Trace URL" in html
    assert "Overall Score" in html
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload[0]["langsmith_run_url"].endswith("/abc")


# ---------------------------------------------------------------------------
# Regression + benchmark
# ---------------------------------------------------------------------------


def _sample_report(
    *,
    question: str,
    score: float,
    latency: float,
    answer: str = "answer",
    tool_passed: bool | None = True,
) -> EvaluationReport:
    tool = None
    if tool_passed is not None:
        tool = ToolValidator().validate(
            [ToolCallExpectation(tool_name="search_docs")],
            (
                [ActualToolCall(tool_name="search_docs", order=0)]
                if tool_passed
                else []
            ),
        )
    return EvaluationReport.build(
        question=question,
        answer=answer,
        expected_answer="expected",
        retrieved_documents=[_doc()],
        metrics=[
            MetricResult(name="faithfulness", score=score, passed=score >= 0.7),
            MetricResult(name="hallucination", score=score, passed=score >= 0.7),
            MetricResult(name="answer_relevancy", score=score, passed=score >= 0.7),
            MetricResult(
                name="contextual_precision",
                score=score,
                passed=score >= 0.7,
            ),
            MetricResult(name="contextual_recall", score=score, passed=score >= 0.7),
        ],
        latency_ms=5.0,
        pass_threshold=0.7,
        rag_latency_ms=latency,
        token_usage={"total_tokens": 100, "prompt_tokens": 60, "completion_tokens": 40},
        estimated_cost_usd=0.001,
        tool_validation=tool,
    )


def test_regression_runner_detects_score_latency_tool_prompt(
    tmp_path: Path,
) -> None:
    previous = [
        _sample_report(question="Q1?", score=0.95, latency=40.0, answer="stable"),
    ]
    current = [
        _sample_report(
            question="Q1?",
            score=0.70,
            latency=80.0,
            answer="changed answer",
            tool_passed=False,
        ),
    ]
    prev_path = tmp_path / "prev.json"
    curr_path = tmp_path / "curr.json"
    prev_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in previous]),
        encoding="utf-8",
    )
    curr_path.write_text(
        json.dumps([r.model_dump(mode="json") for r in current]),
        encoding="utf-8",
    )
    report = RegressionRunner(
        score_drop_threshold=0.05,
        latency_increase_ratio=0.25,
    ).compare_files(prev_path, curr_path)
    assert report.has_regressions is True
    assert report.score_regressions
    assert report.latency_regressions
    assert report.tool_regressions
    assert report.prompt_regressions


def test_benchmark_runner_averages(tmp_path: Path) -> None:
    reports = [
        _sample_report(question="Q1?", score=0.8, latency=40.0),
        _sample_report(question="Q2?", score=1.0, latency=60.0),
    ]
    summary = BenchmarkRunner().summarize(reports)
    assert summary.example_count == 2
    assert summary.average_faithfulness == pytest.approx(0.9)
    assert summary.average_hallucination == pytest.approx(0.9)
    assert summary.average_relevancy == pytest.approx(0.9)
    assert summary.average_context_precision == pytest.approx(0.9)
    assert summary.average_context_recall == pytest.approx(0.9)
    assert summary.average_latency_ms == pytest.approx(50.0)
    assert summary.average_tokens == pytest.approx(100.0)
    assert summary.pass_rate == pytest.approx(1.0)
    path = BenchmarkRunner().write_summary(
        summary,
        tmp_path,
        run_name="bench",
    )
    assert path.is_file()


def test_project_golden_dataset_loads() -> None:
    path = Path("datasets/golden_dataset.json")
    examples = GoldenDatasetLoader().load(path)
    assert len(examples) >= 3
    assert any(e.expected_tools for e in examples)
