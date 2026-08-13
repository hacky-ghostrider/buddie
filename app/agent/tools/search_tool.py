"""Search tool — mock web/search implementation for agent demos.

Production note: replace the mock body with a real search client later
without changing planner contracts or evaluation (Strategy swap).
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

logger = logging.getLogger(__name__)


class SearchTool:
    """Mock external search tool (Sprint 11 stub).

    Returns deterministic mock hits so tests need no live network.
    """

    name = "search"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Return mock search results for ``query``."""
        del context
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        query = str(arguments.get("query") or arguments.get("q") or "")
        try:
            if not query.strip():
                raise ValueError("query must be non-empty")
            hits = [
                {
                    "title": f"Mock result for: {query}",
                    "url": "https://example.com/mock-search",
                    "snippet": (
                        f"This is a mocked search snippet related to '{query}'."
                    ),
                }
            ]
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "search (mock) completed: correlation_id=%s query=%r hits=%d",
                correlation_id,
                query,
                len(hits),
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"query": query},
                output={"results": hits, "provider": "mock"},
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                order=order,
                trace_metadata={"tool": self.name, "mock": True},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "search failed: correlation_id=%s error=%s",
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
                trace_metadata={"tool": self.name, "mock": True},
            )


__all__ = ["SearchTool"]
