"""Evaluation service — runs registered metrics and builds reports.

``EvaluationService`` is the application service for offline / online scoring
of a single ``RAGResponse``. It depends on ``MetricRegistry`` and settings,
never on DeepEval, RAGAS, LangSmith, or Phoenix.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from pydantic import ValidationError

from app.config.settings import Settings, get_settings
from app.evaluation.exceptions import (
    EvaluationDisabledError,
    InvalidEvaluationInputError,
    InvalidEvaluationReportError,
    MetricEvaluationError,
)
from app.evaluation.models import EvaluationContext, MetricResult
from app.evaluation.metrics.base import Metric
from app.evaluation.registry import MetricRegistry
from app.evaluation.report import EvaluationReport
from app.orchestration.models import RAGResponse

logger = logging.getLogger(__name__)


class EvaluationService:
    """Run all enabled metrics against one RAG response and aggregate a report.

    Input:
        - Question
        - Optional expected answer (golden)
        - ``RAGResponse`` from the orchestration layer

    Output:
        - ``EvaluationReport`` with per-metric scores and overall summary

    Args:
        registry: Metric catalog (plugin registry).
        settings: Provides enable flag, pass threshold, and metric timeout.
    """

    def __init__(
        self,
        registry: MetricRegistry,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._settings = settings or get_settings()

    def evaluate(
        self,
        question: str,
        rag_response: RAGResponse,
        expected_answer: str | None = None,
        *,
        expected_sources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Evaluate a RAG response with all enabled registered metrics.

        Args:
            question: Question under evaluation (must be non-blank).
            rag_response: Structured RAG pipeline output.
            expected_answer: Optional golden answer for future semantic metrics.
            expected_sources: Optional expected source identifiers.
            metadata: Optional report-level metadata.

        Returns:
            Aggregated ``EvaluationReport``.

        Raises:
            EvaluationDisabledError: ``ENABLE_EVALUATION`` is false.
            InvalidEvaluationInputError: Blank question or invalid response.
            NoRegisteredMetricsError: Registry has no enabled metrics.
            InvalidEvaluationReportError: Report assembly / validation failed.
            MetricTimeoutError: Propagated only if configured to fail-fast;
                by default timeouts become failed ``MetricResult`` rows.
        """
        if not self._settings.enable_evaluation:
            raise EvaluationDisabledError(
                "Evaluation is disabled (ENABLE_EVALUATION=false)"
            )

        cleaned_question = self._validate_question(question)
        if not isinstance(rag_response, RAGResponse):
            raise InvalidEvaluationInputError(
                f"rag_response must be RAGResponse, got {type(rag_response).__name__}"
            )

        metrics = self._registry.get_enabled_metrics()
        timeout_seconds = self._settings.metric_timeout
        pass_threshold = self._settings.default_pass_threshold

        # Sprint 10.2: prefer the enriched EvaluationContext factory so
        # metrics still see question/answer/docs while automation can attach
        # prompt, tokens, tools, and LangSmith ids on the same object later.
        context = EvaluationContext.from_rag_response(
            question=cleaned_question,
            rag_response=rag_response,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            metadata=dict(metadata or {}),
        )

        logger.info(
            "Evaluation started: question_preview=%r metric_count=%d "
            "timeout_s=%.2f pass_threshold=%.3f",
            cleaned_question[:80],
            len(metrics),
            timeout_seconds,
            pass_threshold,
        )

        started = time.perf_counter()
        results: list[MetricResult] = []
        for metric in metrics:
            result = self._run_metric(metric, context, timeout_seconds)
            results.append(result)
            if result.error:
                logger.warning(
                    "Metric failure: name=%s error=%s latency_ms=%.2f",
                    result.name,
                    result.error,
                    result.latency_ms,
                )
            else:
                logger.info(
                    "Metric executed: name=%s score=%.4f passed=%s latency_ms=%.2f",
                    result.name,
                    result.score,
                    result.passed,
                    result.latency_ms,
                )

        total_ms = (time.perf_counter() - started) * 1000.0

        gen_meta = dict(rag_response.generation_metadata or {})
        token_usage = {
            key: gen_meta[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in gen_meta
        }

        try:
            report = EvaluationReport.build(
                question=cleaned_question,
                answer=rag_response.answer,
                retrieved_documents=list(rag_response.retrieved_documents),
                metrics=results,
                latency_ms=total_ms,
                pass_threshold=pass_threshold,
                expected_answer=expected_answer,
                rag_latency_ms=rag_response.latency.total_ms,
                token_usage=token_usage,
                metadata={
                    **(metadata or {}),
                    "rag_correlation_id": rag_response.correlation_id,
                    "metric_names": [m.name() for m in metrics],
                },
            )
        except (ValidationError, ValueError) as exc:
            logger.exception("Invalid evaluation report assembly")
            raise InvalidEvaluationReportError(
                f"Failed to build EvaluationReport: {exc}"
            ) from exc

        logger.info(
            "Evaluation completed: overall_score=%.4f passed=%s "
            "latency_ms=%.2f metrics=%d",
            report.overall_score,
            report.passed,
            report.latency,
            len(report.metrics),
        )
        return report

    def _run_metric(
        self,
        metric: Metric,
        context: EvaluationContext,
        timeout_seconds: float,
    ) -> MetricResult:
        """Execute one metric with a wall-clock timeout.

        Failures and timeouts become ``MetricResult`` rows with ``score=0``
        so a single broken metric does not abort the whole suite — similar to
        soft-assert aggregation in test frameworks.
        """
        name = metric.name()
        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(metric.evaluate, context)
                result = future.result(timeout=timeout_seconds)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return result.model_copy(update={"latency_ms": latency_ms})
        except FuturesTimeoutError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            message = (
                f"Metric '{name}' exceeded METRIC_TIMEOUT "
                f"({timeout_seconds}s)"
            )
            logger.error(
                "Metric timeout: name=%s timeout_s=%.2f latency_ms=%.2f",
                name,
                timeout_seconds,
                latency_ms,
            )
            return MetricResult(
                name=name,
                score=0.0,
                passed=False,
                details={"timeout_seconds": timeout_seconds},
                error=message,
                latency_ms=latency_ms,
            )
        except MetricEvaluationError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Metric evaluation error: name=%s", name)
            return MetricResult(
                name=name,
                score=0.0,
                passed=False,
                details={},
                error=str(exc),
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001 — isolate metric crashes
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("Unexpected metric failure: name=%s", name)
            return MetricResult(
                name=name,
                score=0.0,
                passed=False,
                details={},
                error=f"Unexpected metric failure: {exc}",
                latency_ms=latency_ms,
            )

    @staticmethod
    def _validate_question(question: str) -> str:
        """Normalize and reject blank questions."""
        cleaned = question.strip()
        if not cleaned:
            raise InvalidEvaluationInputError("question must be a non-empty string")
        return cleaned


def create_default_registry(
    settings: Settings | None = None,
) -> MetricRegistry:
    """Build a registry with placeholders and optional DeepEval adapters.

    Args:
        settings: When ``enable_deepeval`` is true, registers DeepEval metrics.

    Returns:
        ``MetricRegistry`` ready for ``EvaluationService``.
    """
    from app.evaluation.metrics import AnswerLengthMetric, ContextCountMetric

    cfg = settings or get_settings()
    registry = MetricRegistry()
    registry.register(AnswerLengthMetric())
    registry.register(ContextCountMetric())

    if cfg.enable_deepeval:
        try:
            from app.evaluation.deepeval import create_deepeval_metrics
        except ImportError:
            logger.warning(
                "ENABLE_DEEPEVAL=true but deepeval package is unavailable; "
                "continuing with placeholder metrics only"
            )
        else:
            for adapter in create_deepeval_metrics(
                pass_threshold=cfg.default_pass_threshold,
            ):
                registry.register(adapter)
            logger.info("DeepEval metrics registered in default registry")
    else:
        logger.info("DeepEval disabled; placeholder metrics only")

    return registry


__all__ = [
    "EvaluationService",
    "create_default_registry",
]
