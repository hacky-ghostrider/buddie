"""RAG-backed agent tools — reuse ``RAGService`` (no duplicated RAG logic).

Exposes two tool names that match the canonical Sprint 10.2 contracts:

* ``search_docs`` — retrieve grounded evidence via ``RAGService.query``
* ``summarize`` — return / produce a summary from prior RAG evidence

Together they are the agent-facing surface of the RAG platform. The full
retrieve→prompt→LLM pipeline remains owned by ``RAGService``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionStatus,
)
from app.orchestration.models import RAGRequest, RAGResponse
from app.orchestration.rag_service import RAGService

logger = logging.getLogger(__name__)

SEARCH_DOCS_TOOL = "search_docs"
SUMMARIZE_TOOL = "summarize"
SEARCH_COMPANY_POLICY_TOOL = "search_company_policy"


class SearchDocsTool:
    """Agent tool that searches the knowledge base via ``RAGService``.

    Args:
        rag_service: Injected RAG orchestrator (required).
    """

    name = SEARCH_DOCS_TOOL

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service
        self._last_response: RAGResponse | None = None

    @property
    def last_response(self) -> RAGResponse | None:
        """Most recent ``RAGResponse`` (shared with ``SummarizeTool``)."""
        return self._last_response

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Run retrieval+generation via RAG and return document evidence."""
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        query = str(arguments.get("query") or (context or {}).get("question") or "")
        try:
            response = self._rag_service.query(RAGRequest(question=query))
            self._last_response = response
            if context is not None:
                context["last_rag_response"] = response
                context["retrieved_documents"] = list(response.retrieved_documents)
            latency_ms = (time.perf_counter() - started) * 1000.0
            docs_payload = [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": doc.score,
                    "metadata": dict(doc.metadata or {}),
                }
                for doc in response.retrieved_documents
            ]
            logger.info(
                "search_docs completed: correlation_id=%s docs=%d latency_ms=%.1f",
                correlation_id or response.correlation_id,
                len(docs_payload),
                latency_ms,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"query": query},
                output={
                    "documents": docs_payload,
                    "answer_preview": response.answer,
                    "correlation_id": response.correlation_id,
                    "retrieval_metadata": dict(response.retrieval_metadata),
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                order=order,
                trace_metadata={
                    "rag_correlation_id": response.correlation_id,
                    "tool": self.name,
                },
            )
        except Exception as exc:  # noqa: BLE001 — surface as ToolExecution
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "search_docs failed: correlation_id=%s error=%s",
                correlation_id,
                exc,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"query": query},
                output=None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.FAILED,
                error=str(exc),
                order=order,
                trace_metadata={"tool": self.name},
            )


class SummarizeTool:
    """Agent tool that summarizes document evidence from prior RAG search.

    Prefers the cached ``RAGResponse`` from ``SearchDocsTool``. When absent,
    falls back to ``RAGService.query`` using the agent question.

    Args:
        rag_service: Injected RAG orchestrator.
        search_docs_tool: Optional sibling tool that holds the last response.
    """

    name = SUMMARIZE_TOOL

    def __init__(
        self,
        rag_service: RAGService,
        search_docs_tool: SearchDocsTool | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._search_docs_tool = search_docs_tool

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Produce a summary string grounded in RAG evidence."""
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        document = str(arguments.get("document") or "employee_handbook.pdf")
        ctx = context or {}

        try:
            response = self._resolve_response(ctx, correlation_id=correlation_id)
            summary = response.answer
            if ctx is not None:
                ctx["last_rag_response"] = response
                ctx["summary"] = summary
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "summarize completed: correlation_id=%s document=%s latency_ms=%.1f",
                correlation_id or response.correlation_id,
                document,
                latency_ms,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"document": document},
                output={
                    "summary": summary,
                    "document": document,
                    "model": (response.generation_metadata or {}).get("model"),
                    "prompt": (response.generation_metadata or {}).get("prompt"),
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                order=order,
                trace_metadata={
                    "rag_correlation_id": response.correlation_id,
                    "tool": self.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "summarize failed: correlation_id=%s error=%s",
                correlation_id,
                exc,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"document": document},
                output=None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.FAILED,
                error=str(exc),
                order=order,
                trace_metadata={"tool": self.name},
            )

    def _resolve_response(
        self,
        context: dict[str, Any],
        *,
        correlation_id: str | None,
    ) -> RAGResponse:
        """Reuse cached RAG response or run ``RAGService`` once."""
        cached = context.get("last_rag_response")
        if isinstance(cached, RAGResponse):
            return cached
        if self._search_docs_tool and self._search_docs_tool.last_response is not None:
            return self._search_docs_tool.last_response

        question = str(context.get("question") or "")
        logger.info(
            "summarize falling back to RAGService.query: correlation_id=%s",
            correlation_id,
        )
        return self._rag_service.query(RAGRequest(question=question))


class SearchCompanyPolicyTool:
    """Explicit policy-retrieval tool for multi-tool workflows.

    Reuses ``RAGService`` — does not duplicate retrieval/generation logic.
    """

    name = SEARCH_COMPANY_POLICY_TOOL

    def __init__(self, rag_service: RAGService) -> None:
        self._rag_service = rag_service

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Retrieve company policy excerpts via the shared RAG pipeline."""
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        query = str(
            arguments.get("query")
            or (context or {}).get("question")
            or ""
        )
        try:
            response = self._rag_service.query(RAGRequest(question=query))
            if context is not None:
                context["last_rag_response"] = response
                context["retrieved_documents"] = list(response.retrieved_documents)
                context["summary"] = response.answer
            latency_ms = (time.perf_counter() - started) * 1000.0
            sources = []
            for doc in response.retrieved_documents:
                meta = dict(doc.metadata or {})
                label = meta.get("source") or meta.get("file_name") or doc.id
                if label:
                    sources.append(str(label))
            logger.info(
                "search_company_policy completed: correlation_id=%s docs=%d",
                correlation_id or response.correlation_id,
                len(response.retrieved_documents),
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"query": query},
                output={
                    "query": query,
                    "summary": response.answer,
                    "sources": sources,
                    "document_count": len(response.retrieved_documents),
                    "correlation_id": response.correlation_id,
                },
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                order=order,
                trace_metadata={
                    "rag_correlation_id": response.correlation_id,
                    "tool": self.name,
                },
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "search_company_policy failed: correlation_id=%s error=%s",
                correlation_id,
                exc,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"query": query},
                output=None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.FAILED,
                error=str(exc),
                order=order,
                trace_metadata={"tool": self.name},
            )


class RAGToolBundle:
    """Factory that wires RAG-backed tools onto one ``RAGService``.

    RAG remains *one platform capability*; the agent sees contract-aligned
    tool names so evaluation can assert retrieve-then-summarize order.
    """

    def __init__(self, rag_service: RAGService) -> None:
        self.search_docs = SearchDocsTool(rag_service)
        self.summarize = SummarizeTool(rag_service, self.search_docs)
        self.search_company_policy = SearchCompanyPolicyTool(rag_service)

    def tools(self) -> list[SearchDocsTool | SummarizeTool | SearchCompanyPolicyTool]:
        """Return RAG tools for registry registration."""
        return [self.search_docs, self.summarize, self.search_company_policy]


__all__ = [
    "SEARCH_DOCS_TOOL",
    "SUMMARIZE_TOOL",
    "SEARCH_COMPANY_POLICY_TOOL",
    "SearchDocsTool",
    "SummarizeTool",
    "SearchCompanyPolicyTool",
    "RAGToolBundle",
]
