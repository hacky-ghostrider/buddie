"""Tracing service — application façade over ``Tracer`` strategies.

WHY a service
-------------
Callers (evaluation automation, RAG hooks) should not know whether
LangSmith, Phoenix, or a no-op tracer is active. ``TracingService``
reads settings, selects the adapter, and builds ``TraceSpanData`` from
RAG / evaluation artifacts — classic façade + DI.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import Settings, get_settings
from app.evaluation.report import EvaluationReport
from app.orchestration.models import RAGResponse
from app.tracing.base import NoOpTracer, TraceRecord, TraceSpanData, Tracer
from app.tracing.langsmith_adapter import LangSmithTracer

logger = logging.getLogger(__name__)


class TracingService:
    """Record RAG + evaluation executions via an injectable ``Tracer``.

    Args:
        tracer: Concrete tracer strategy.
        settings: Optional settings (for factory helpers).
    """

    def __init__(
        self,
        tracer: Tracer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._tracer = tracer if tracer is not None else create_tracer(self._settings)

    @property
    def tracer(self) -> Tracer:
        """Return the active tracer strategy."""
        return self._tracer

    def trace_rag_evaluation(
        self,
        *,
        rag_response: RAGResponse,
        evaluation_report: EvaluationReport | None = None,
        prompt: dict[str, Any] | str | None = None,
        tool_validation: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """Build a span from RAG (+ optional eval) and record it.

        Args:
            rag_response: Pipeline output (answer, chunks, latency, tokens).
            evaluation_report: Optional metric report to attach.
            prompt: Optional prompt payload from generation metadata.
            tool_validation: Optional tool-validation summary.
            extra_metadata: Optional extra metadata merged into the span.

        Returns:
            ``TraceRecord`` from the underlying tracer.
        """
        gen_meta = dict(rag_response.generation_metadata or {})
        tokens = {
            "prompt_tokens": gen_meta.get("prompt_tokens"),
            "completion_tokens": gen_meta.get("completion_tokens"),
            "total_tokens": gen_meta.get("total_tokens"),
        }
        model = gen_meta.get("model")
        resolved_prompt = prompt if prompt is not None else gen_meta.get("prompt")

        evaluation_results: dict[str, Any] = {}
        if evaluation_report is not None:
            evaluation_results["overall_score"] = evaluation_report.overall_score
            evaluation_results["passed"] = evaluation_report.passed
            evaluation_results["metrics"] = [
                {
                    "name": m.name,
                    "score": m.score,
                    "passed": m.passed,
                    "error": m.error,
                }
                for m in evaluation_report.metrics
            ]
        if tool_validation is not None:
            evaluation_results["tool_validation"] = tool_validation

        span = TraceSpanData(
            question=rag_response.question,
            retrieved_chunks=[d.text for d in rag_response.retrieved_documents],
            prompt=resolved_prompt,
            model=str(model) if model else None,
            tokens={k: v for k, v in tokens.items() if v is not None},
            latency_ms=rag_response.latency.total_ms,
            answer=rag_response.answer,
            evaluation_results=evaluation_results,
            metadata={
                "correlation_id": rag_response.correlation_id,
                **(extra_metadata or {}),
            },
        )
        record = self._tracer.record(span)
        logger.info(
            "Trace recorded: enabled=%s run_id=%s url=%s",
            record.enabled,
            record.run_id,
            record.run_url,
        )
        return record


def create_tracer(settings: Settings | None = None) -> Tracer:
    """Factory: LangSmith when enabled, otherwise ``NoOpTracer``.

    Args:
        settings: Application settings.

    Returns:
        Concrete ``Tracer`` implementation.
    """
    cfg = settings or get_settings()
    if not cfg.enable_langsmith:
        logger.info("LangSmith disabled; using NoOpTracer")
        return NoOpTracer()
    logger.info(
        "LangSmith enabled; project=%s",
        cfg.langsmith_project,
    )
    return LangSmithTracer(
        project_name=cfg.langsmith_project,
        api_key=cfg.langsmith_api_key or None,
    )


__all__ = [
    "TracingService",
    "create_tracer",
]
