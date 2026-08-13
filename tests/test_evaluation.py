"""Tests for the Sprint 9 evaluation framework.

Evaluation frameworks should be **deterministic**: the same question,
``RAGResponse``, and metric configuration must produce the same scores.
Non-determinism (random sampling, live LLM judges without seeds) makes
regression comparison and CI gates unreliable — like flaky Selenium tests.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.evaluation.evaluator import EvaluationService, create_default_registry
from app.evaluation.exceptions import (
    EvaluationDisabledError,
    InvalidEvaluationInputError,
    InvalidEvaluationReportError,
    MetricEvaluationError,
    MetricNotFoundError,
    MetricRegistrationError,
    NoRegisteredMetricsError,
)
from app.evaluation.metrics import AnswerLengthMetric, ContextCountMetric
from app.evaluation.metrics.base import Metric
from app.evaluation.models import EvaluationContext, GoldenExample, MetricResult
from app.evaluation.registry import MetricRegistry
from app.evaluation.report import EvaluationReport
from app.orchestration.models import LatencyBreakdown, RAGResponse
from app.retrieval.models import RetrievedDocument


def _doc(
    text: str = "Recursive chunking splits on separators.",
    *,
    doc_id: str = "c1",
    score: float = 0.91,
) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        text=text,
        metadata={"file_name": "guide.pdf"},
        score=score,
    )


def _rag_response(
    *,
    question: str = "What is chunking?",
    answer: str = "Chunking splits long text into overlapping windows.",
    documents: list[RetrievedDocument] | None = None,
) -> RAGResponse:
    return RAGResponse(
        question=question,
        answer=answer,
        retrieved_documents=documents if documents is not None else [_doc()],
        retrieval_metadata={},
        generation_metadata={},
        latency=LatencyBreakdown(
            retrieval_ms=10.0,
            prompt_build_ms=2.0,
            llm_ms=50.0,
            total_ms=62.0,
        ),
        correlation_id="corr-eval-1",
    )


@pytest.fixture
def eval_settings() -> Settings:
    """Settings with evaluation enabled and a moderate pass threshold."""
    return Settings(
        app_env="test",
        enable_evaluation=True,
        enable_deepeval=False,
        enable_langsmith=False,
        default_pass_threshold=0.7,
        metric_timeout=5.0,
        log_level="WARNING",
    )


class _FixedScoreMetric(Metric):
    """Deterministic mock metric for registry / aggregation tests."""

    def __init__(
        self,
        metric_name: str,
        score: float,
        *,
        should_fail: bool = False,
    ) -> None:
        self._metric_name = metric_name
        self._score = score
        self._should_fail = should_fail

    def name(self) -> str:
        return self._metric_name

    def description(self) -> str:
        return f"Fixed score metric {self._metric_name}"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        if self._should_fail:
            raise MetricEvaluationError(f"{self._metric_name} boom")
        return MetricResult(
            name=self._metric_name,
            score=self._score,
            passed=self._score >= 0.7,
            details={"question": context.question},
        )


class _SlowMetric(Metric):
    """Metric that sleeps longer than the configured timeout."""

    def name(self) -> str:
        return "slow_metric"

    def description(self) -> str:
        return "Intentionally slow metric"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        time.sleep(2.0)
        return MetricResult(
            name=self.name(),
            score=1.0,
            passed=True,
            details={},
        )


# ---------------------------------------------------------------------------
# Metric registration
# ---------------------------------------------------------------------------


def test_metric_registration_and_retrieval() -> None:
    registry = MetricRegistry()
    metric = AnswerLengthMetric(min_length=10)
    registry.register(metric)

    assert registry.list_registered() == ["answer_length"]
    assert registry.get("answer_length") is metric
    assert registry.is_enabled("answer_length") is True


def test_metric_enable_disable() -> None:
    registry = MetricRegistry([AnswerLengthMetric(), ContextCountMetric()])
    registry.disable("answer_length")

    assert registry.list_enabled() == ["context_count"]
    registry.enable("answer_length")
    assert set(registry.list_enabled()) == {"answer_length", "context_count"}


def test_metric_registration_rejects_blank_name() -> None:
    class _BlankNameMetric(Metric):
        def name(self) -> str:
            return "  "

        def description(self) -> str:
            return "blank"

        def evaluate(self, context: EvaluationContext) -> MetricResult:
            return MetricResult(name="x", score=1.0, passed=True)

    registry = MetricRegistry()
    with pytest.raises(MetricRegistrationError):
        registry.register(_BlankNameMetric())


def test_metric_not_found() -> None:
    registry = MetricRegistry()
    with pytest.raises(MetricNotFoundError):
        registry.get("missing")


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


def test_empty_registry_raises(eval_settings: Settings) -> None:
    registry = MetricRegistry()
    service = EvaluationService(registry=registry, settings=eval_settings)

    with pytest.raises(NoRegisteredMetricsError):
        service.evaluate("What is chunking?", _rag_response())


def test_all_disabled_metrics_raise(eval_settings: Settings) -> None:
    registry = MetricRegistry([AnswerLengthMetric()])
    registry.disable("answer_length")
    service = EvaluationService(registry=registry, settings=eval_settings)

    with pytest.raises(NoRegisteredMetricsError):
        service.evaluate("What is chunking?", _rag_response())


# ---------------------------------------------------------------------------
# Metric execution + report generation
# ---------------------------------------------------------------------------


def test_metric_execution_and_report(eval_settings: Settings) -> None:
    registry = MetricRegistry(
        [
            AnswerLengthMetric(min_length=10, pass_threshold=0.7),
            ContextCountMetric(min_documents=1, pass_threshold=0.7),
        ]
    )
    service = EvaluationService(registry=registry, settings=eval_settings)
    response = _rag_response()

    report = service.evaluate(
        response.question,
        response,
        expected_answer="Chunking splits text.",
    )

    assert isinstance(report, EvaluationReport)
    assert report.question == response.question
    assert report.answer == response.answer
    assert len(report.retrieved_documents) == 1
    assert len(report.metrics) == 2
    assert report.overall_score == pytest.approx(1.0)
    assert report.passed is True
    assert report.latency >= 0.0
    assert report.evaluation_time is not None
    names = {item.name for item in report.metrics}
    assert names == {"answer_length", "context_count"}


def test_overall_score_is_mean_of_metric_scores(eval_settings: Settings) -> None:
    registry = MetricRegistry(
        [
            _FixedScoreMetric("a", 1.0),
            _FixedScoreMetric("b", 0.4),
        ]
    )
    # Lower threshold so we can assert score independently of pass flag.
    settings = eval_settings.model_copy(update={"default_pass_threshold": 0.5})
    service = EvaluationService(registry=registry, settings=settings)

    report = service.evaluate("Q?", _rag_response(question="Q?", answer="short"))

    assert report.overall_score == pytest.approx(0.7)
    assert report.passed is True


def test_failed_metric_records_zero_score(eval_settings: Settings) -> None:
    registry = MetricRegistry(
        [
            _FixedScoreMetric("ok", 1.0),
            _FixedScoreMetric("bad", 0.0, should_fail=True),
        ]
    )
    service = EvaluationService(registry=registry, settings=eval_settings)

    report = service.evaluate("Q?", _rag_response(question="Q?"))

    by_name = {item.name: item for item in report.metrics}
    assert by_name["ok"].score == 1.0
    assert by_name["ok"].error is None
    assert by_name["bad"].score == 0.0
    assert by_name["bad"].passed is False
    assert by_name["bad"].error is not None
    assert "boom" in by_name["bad"].error
    assert report.overall_score == pytest.approx(0.5)
    assert report.passed is False


def test_metric_timeout_becomes_failed_result() -> None:
    settings = Settings(
        app_env="test",
        enable_evaluation=True,
        default_pass_threshold=0.7,
        metric_timeout=0.2,
        log_level="WARNING",
    )
    registry = MetricRegistry([_SlowMetric()])
    service = EvaluationService(registry=registry, settings=settings)

    report = service.evaluate("Q?", _rag_response(question="Q?"))

    assert len(report.metrics) == 1
    assert report.metrics[0].score == 0.0
    assert report.metrics[0].passed is False
    assert report.metrics[0].error is not None
    assert "METRIC_TIMEOUT" in report.metrics[0].error


# ---------------------------------------------------------------------------
# Placeholder metrics
# ---------------------------------------------------------------------------


def test_answer_length_metric_scores_short_answer() -> None:
    metric = AnswerLengthMetric(min_length=100, pass_threshold=0.7)
    context = EvaluationContext(
        question="What is RAG?",
        answer="Too short",
        retrieved_documents=[],
    )
    result = metric.evaluate(context)

    assert result.name == "answer_length"
    assert result.score < 0.7
    assert result.passed is False
    assert result.details["answer_length"] == len("Too short")


def test_context_count_metric_scores_empty_retrieval() -> None:
    metric = ContextCountMetric(min_documents=2, pass_threshold=0.7)
    context = EvaluationContext(
        question="What is RAG?",
        answer="An answer with enough characters here.",
        retrieved_documents=[_doc()],
    )
    result = metric.evaluate(context)

    assert result.name == "context_count"
    assert result.score == pytest.approx(0.5)
    assert result.passed is False


def test_create_default_registry_registers_placeholders() -> None:
    registry = create_default_registry(
        Settings(
            app_env="test",
            enable_deepeval=False,
            log_level="WARNING",
        )
    )
    assert set(registry.list_enabled()) == {"answer_length", "context_count"}


# ---------------------------------------------------------------------------
# Report generation / invalid report
# ---------------------------------------------------------------------------


def test_report_build_rejects_inconsistent_passed_flag() -> None:
    from datetime import datetime, timezone

    with pytest.raises(ValidationError):
        EvaluationReport(
            question="Q?",
            answer="A",
            retrieved_documents=[],
            metrics=[],
            overall_score=0.9,
            passed=False,  # inconsistent with 0.9 >= 0.7
            evaluation_time=datetime.now(timezone.utc),
            latency=1.0,
            pass_threshold=0.7,
        )


def test_invalid_report_from_service_is_wrapped(
    eval_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force report assembly failure via a patched build method."""
    registry = MetricRegistry([_FixedScoreMetric("a", 1.0)])
    service = EvaluationService(registry=registry, settings=eval_settings)

    def _boom(**kwargs: object) -> EvaluationReport:
        raise ValueError("synthetic report failure")

    monkeypatch.setattr(EvaluationReport, "build", classmethod(lambda cls, **kw: _boom(**kw)))

    with pytest.raises(InvalidEvaluationReportError):
        service.evaluate("Q?", _rag_response(question="Q?"))


# ---------------------------------------------------------------------------
# Input / enable guards
# ---------------------------------------------------------------------------


def test_blank_question_rejected(eval_settings: Settings) -> None:
    service = EvaluationService(
        registry=MetricRegistry([AnswerLengthMetric()]),
        settings=eval_settings,
    )
    with pytest.raises(InvalidEvaluationInputError):
        service.evaluate("   ", _rag_response())


def test_evaluation_disabled(eval_settings: Settings) -> None:
    settings = eval_settings.model_copy(update={"enable_evaluation": False})
    service = EvaluationService(
        registry=MetricRegistry([AnswerLengthMetric()]),
        settings=settings,
    )
    with pytest.raises(EvaluationDisabledError):
        service.evaluate("Q?", _rag_response(question="Q?"))


# ---------------------------------------------------------------------------
# Golden dataset model
# ---------------------------------------------------------------------------


def test_golden_example_model() -> None:
    example = GoldenExample(
        question="What is recursive chunking?",
        expected_answer="Splitting on hierarchical separators.",
        expected_sources=["guide.pdf"],
        expected_tools=["search_docs"],
        expected_tool_arguments=[{"query": "chunking"}],
        expected_tool_order=["search_docs"],
        tags=["chunking", "ingestion"],
        difficulty="medium",
        category="rag",
    )
    assert example.difficulty == "medium"
    assert example.tags == ["chunking", "ingestion"]
    assert example.expected_tools == ["search_docs"]
    assert example.category == "rag"


def test_golden_example_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        GoldenExample(question="  ", expected_answer="answer")


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


def test_settings_reject_invalid_pass_threshold() -> None:
    with pytest.raises(ValidationError):
        Settings(default_pass_threshold=1.5)


def test_settings_reject_non_positive_metric_timeout() -> None:
    with pytest.raises(ValidationError):
        Settings(metric_timeout=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_evaluation_is_deterministic(eval_settings: Settings) -> None:
    registry = MetricRegistry(
        [
            AnswerLengthMetric(min_length=10),
            ContextCountMetric(min_documents=1),
        ]
    )
    service = EvaluationService(registry=registry, settings=eval_settings)
    response = _rag_response()

    first = service.evaluate(response.question, response)
    second = service.evaluate(response.question, response)

    assert first.overall_score == second.overall_score
    assert [m.score for m in first.metrics] == [m.score for m in second.metrics]
    assert first.passed == second.passed


def test_mock_metric_receives_context(eval_settings: Settings) -> None:
    mock_metric = MagicMock(spec=Metric)
    mock_metric.name.return_value = "mock_metric"
    mock_metric.description.return_value = "mock"
    mock_metric.evaluate.return_value = MetricResult(
        name="mock_metric",
        score=1.0,
        passed=True,
    )

    registry = MetricRegistry()
    registry.register(mock_metric)
    service = EvaluationService(registry=registry, settings=eval_settings)
    response = _rag_response(answer="Deterministic answer text.")

    report = service.evaluate(
        response.question,
        response,
        expected_answer="Expected",
    )

    assert report.metrics[0].name == "mock_metric"
    mock_metric.evaluate.assert_called_once()
    ctx = mock_metric.evaluate.call_args.args[0]
    assert isinstance(ctx, EvaluationContext)
    assert ctx.expected_answer == "Expected"
    assert ctx.answer == response.answer
