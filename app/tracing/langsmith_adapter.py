"""LangSmith tracer adapter — keeps LangSmith SDK behind ``Tracer``.

WHY LangSmith here
-----------------
LangSmith is an observability / tracing product: run trees, prompts,
token usage, latency, and dataset experiments. It is **not** a
replacement for DeepEval faithfulness scores. We use it to debug *what*
happened; DeepEval judges *how good* the answer was.

HOW
---
``Client.create_run`` + ``update_run`` (or injectable client) so unit
tests never hit the network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from app.tracing.base import TraceRecord, TraceSpanData, Tracer

logger = logging.getLogger(__name__)


class LangSmithClientLike(Protocol):
    """Minimal client surface used by the adapter (for DI / mocks)."""

    def create_run(self, **kwargs: Any) -> Any:
        """Create a run and return an object or id."""

    def update_run(self, run_id: Any, **kwargs: Any) -> Any:
        """Finalize / update a run."""


class LangSmithTracer(Tracer):
    """Record RAG evaluation spans to LangSmith.

    Args:
        project_name: LangSmith project name.
        api_key: API key (optional if client injected / env configured).
        client: Optional injectable client (tests).
        base_url: Optional UI base used to build ``run_url``.
    """

    def __init__(
        self,
        *,
        project_name: str = "rag-evaluation",
        api_key: str | None = None,
        client: LangSmithClientLike | None = None,
        base_url: str = "https://smith.langchain.com",
    ) -> None:
        self._project_name = project_name.strip() or "rag-evaluation"
        self._api_key = (api_key or "").strip() or None
        self._client = client
        self._base_url = base_url.rstrip("/")

    def record(self, span: TraceSpanData) -> TraceRecord:
        """Create and finalize a LangSmith run for ``span``.

        Args:
            span: Execution payload.

        Returns:
            ``TraceRecord`` with run / trace ids and URL when successful.
            On failure, returns ``enabled=False`` with error in logs
            (tracing must not break evaluation).
        """
        try:
            client = self._resolve_client()
            run_id = str(uuid4())
            start_time = datetime.now(timezone.utc)
            inputs = {
                "question": span.question,
                "retrieved_chunks": span.retrieved_chunks,
                "prompt": span.prompt,
                "model": span.model,
            }
            outputs = {
                "answer": span.answer,
                "tokens": span.tokens,
                "latency_ms": span.latency_ms,
                "evaluation_results": span.evaluation_results,
                "metadata": span.metadata,
            }
            client.create_run(
                id=run_id,
                name="rag_evaluation",
                run_type="chain",
                inputs=inputs,
                project_name=self._project_name,
                start_time=start_time,
                extra={"metadata": span.metadata},
            )
            end_time = datetime.now(timezone.utc)
            client.update_run(
                run_id,
                outputs=outputs,
                end_time=end_time,
            )
            url = f"{self._base_url}/runs/{run_id}"
            logger.info(
                "LangSmith run recorded: run_id=%s project=%s",
                run_id,
                self._project_name,
            )
            return TraceRecord(
                run_id=run_id,
                trace_id=run_id,
                run_url=url,
                project=self._project_name,
                recorded_at=end_time,
                enabled=True,
            )
        except Exception:  # noqa: BLE001 — never fail the eval pipeline
            logger.exception("LangSmith tracing failed; continuing without trace")
            return TraceRecord(
                run_id=None,
                trace_id=None,
                run_url=None,
                project=self._project_name,
                enabled=False,
            )

    def _resolve_client(self) -> LangSmithClientLike:
        """Return injected client or construct LangSmith ``Client``."""
        if self._client is not None:
            return self._client
        try:
            from langsmith import Client
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "langsmith is not installed; inject a client or add dependency"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return Client(**kwargs)


__all__ = ["LangSmithTracer", "LangSmithClientLike"]
