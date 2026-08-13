"""Direct vs MCP tool adapter boundary for Buddie.

Planner → ToolRouter → ToolRegistry → AgentTool
                                      ├─ Direct tools (existing)
                                      └─ MCP adapters (Sprint 15)

LangGraph / RuleBasedPlanner / PlannerOutput remain unchanged.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal

from app.agent.tools.base import AgentTool, ToolRegistry
from app.agent.tools.employee_tools import (
    verified_employee_id_from_context,
)
from app.agent.tools import build_default_tool_registry
from app.employees.service import EmployeeService
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionMetrics,
    ToolExecutionStatus,
)
from app.mcp.client import BuddieMcpClient, McpClientError
from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES
from app.mcp.server import build_buddie_mcp_server
from app.orchestration.rag_service import RAGService

logger = logging.getLogger(__name__)

MCP_PRIMARY_TOOL_NAMES = frozenset(EXPECTED_MCP_TOOL_NAMES)
ToolMode = Literal["direct", "mcp"]


def _safe_summary(data: dict[str, Any]) -> str:
    if not data:
        return "empty"
    if "leave_balance" in data:
        lb = data.get("leave_balance") or {}
        return (
            "leave_balance "
            f"vacation={lb.get('vacation')} sick={lb.get('sick')} "
            f"personal={lb.get('personal')}"
        )
    if "eligible" in data:
        return f"eligible={data.get('eligible')} days={data.get('requested_days')}"
    if "answer" in data:
        answer = str(data.get("answer") or "")
        return f"policy answer chars={len(answer)}"
    if "request_id" in data:
        return f"leave_request id={data.get('request_id')} status={data.get('status')}"
    if "holidays" in data:
        return f"holidays count={data.get('count', len(data.get('holidays') or []))}"
    if "manager" in data or "manager_name" in data:
        return "manager_information"
    keys = ", ".join(list(data.keys())[:5])
    return f"dict keys=[{keys}]"


class McpAgentTool:
    """``AgentTool`` adapter that executes via ``BuddieMcpClient``."""

    def __init__(self, name: str, client: BuddieMcpClient) -> None:
        self._name = name
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        args = dict(arguments or {})
        # Never forward spoofable employee_id to MCP protected tools.
        args.pop("employee_id", None)
        args.pop("id", None)

        verified = verified_employee_id_from_context(context)
        leave_confirmed = bool(args.get("confirmed"))
        if context is not None and context.get("leave_request_confirmed"):
            leave_confirmed = True

        mcp_latency_ms = 0.0
        try:
            call_started = time.perf_counter()
            payload = self._client.call_tool(
                self._name,
                args,
                verified_employee_id=verified,
                leave_request_confirmed=leave_confirmed,
                correlation_id=correlation_id,
            )
            mcp_latency_ms = (time.perf_counter() - call_started) * 1000.0
        except McpClientError as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            finished_at = datetime.now(timezone.utc)
            status = ToolExecutionStatus.FAILED
            metrics = ToolExecutionMetrics.from_latency(
                execution_time_ms=latency_ms,
                status=status,
                failure_reason=exc.message,
                started_at=started_at,
                finished_at=finished_at,
            )
            return ToolExecution(
                tool_name=self._name,
                arguments=args,
                output={
                    "ok": False,
                    "error": exc.message,
                    "error_code": exc.code,
                },
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                status=status,
                error=exc.message,
                order=order,
                metrics=metrics,
                trace_metadata={
                    "tool": self._name,
                    "protocol": "MCP",
                    "mcp_transport": self._client.transport,
                    "mcp_connected": self._client.connected,
                    "mcp_latency_ms": latency_ms,
                    "error_code": exc.code,
                },
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        finished_at = datetime.now(timezone.utc)
        ok = bool(payload.get("ok", True)) and not payload.get("error")
        # Bridge returns ok=False with error fields for auth / validation.
        if payload.get("ok") is False:
            ok = False
        data = payload.get("data")
        if not isinstance(data, dict):
            # Some successful payloads may be the service dict itself.
            data = {
                k: v
                for k, v in payload.items()
                if k not in {"ok", "tool", "error", "error_code"}
            }
            if payload.get("ok") is True and "data" in payload:
                data = payload.get("data") or {}

        # Prefer nested service payload for router answer derivation.
        output: dict[str, Any]
        if isinstance(payload.get("data"), dict):
            output = dict(payload["data"])
        else:
            output = dict(data) if isinstance(data, dict) else {}

        error = None if ok else str(payload.get("error") or "MCP tool failed")
        status = (
            ToolExecutionStatus.SUCCESS if ok else ToolExecutionStatus.FAILED
        )
        metrics = ToolExecutionMetrics.from_latency(
            execution_time_ms=latency_ms,
            status=status,
            failure_reason=error,
            started_at=started_at,
            finished_at=finished_at,
        )

        if context is not None and ok:
            context["employee_tool_result"] = output
            if self._name == "search_company_policy":
                summary = output.get("summary") or output.get("answer")
                if summary:
                    context["summary"] = summary
                docs = output.get("documents")
                if isinstance(docs, list):
                    context["retrieved_documents"] = docs

        logger.info(
            "MCP tool %s status=%s mcp_latency_ms=%.1f",
            self._name,
            status.value,
            mcp_latency_ms,
        )
        return ToolExecution(
            tool_name=self._name,
            arguments=args,
            output=output if ok else {
                **output,
                "ok": False,
                "error": error,
                "error_code": payload.get("error_code"),
            },
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=latency_ms,
            status=status,
            error=error,
            order=order,
            metrics=metrics,
            trace_metadata={
                "tool": self._name,
                "protocol": "MCP",
                "mcp_transport": self._client.transport,
                "mcp_connected": True,
                "mcp_latency_ms": round(mcp_latency_ms, 3),
                "result_summary": _safe_summary(output) if ok else error,
                "error_code": payload.get("error_code"),
            },
        )


def apply_mcp_adapters(
    registry: ToolRegistry,
    client: BuddieMcpClient,
    *,
    tool_names: frozenset[str] | None = None,
) -> ToolRegistry:
    """Replace selected registry tools with MCP-backed adapters."""
    names = tool_names or MCP_PRIMARY_TOOL_NAMES
    for name in names:
        registry.register(McpAgentTool(name, client))
    return registry


def build_tool_registry(
    rag_service: RAGService,
    *,
    employee_service: EmployeeService | None = None,
    tool_mode: ToolMode = "direct",
    mcp_transport: str = "memory",
    mcp_server_command: list[str] | None = None,
    mcp_server_url: str | None = None,
    mcp_timeout_seconds: float = 30.0,
    mcp_client: BuddieMcpClient | None = None,
) -> tuple[ToolRegistry, BuddieMcpClient | None]:
    """Build Direct or MCP-mode registry without changing planner/router."""
    registry = build_default_tool_registry(
        rag_service,
        employee_service=employee_service,
    )
    if tool_mode != "mcp":
        return registry, None

    client = mcp_client
    if client is None:
        transport = (mcp_transport or "memory").strip().lower()
        try:
            if transport == "memory":
                server = build_buddie_mcp_server(
                    employee_service=employee_service,
                    rag_service=rag_service,
                )
                client = BuddieMcpClient(
                    transport="memory",
                    server=server,
                    timeout_seconds=mcp_timeout_seconds,
                )
            elif transport == "stdio":
                client = BuddieMcpClient(
                    transport="stdio",
                    server_command=mcp_server_command,
                    timeout_seconds=mcp_timeout_seconds,
                )
            elif transport == "http":
                client = BuddieMcpClient(
                    transport="http",
                    server_url=mcp_server_url,
                    timeout_seconds=mcp_timeout_seconds,
                )
            else:
                logger.warning(
                    "Unknown mcp_transport=%s — falling back to direct tools",
                    mcp_transport,
                )
                return registry, None
        except McpClientError as exc:
            logger.error(
                "MCP client misconfigured (%s) — falling back to direct tools",
                exc.message,
            )
            return registry, None

    try:
        client.connect()
        client.ensure_expected_tools()
    except McpClientError as exc:
        logger.error(
            "MCP mode unavailable (%s) — falling back to direct tools",
            exc.message,
        )
        return registry, client

    apply_mcp_adapters(registry, client)
    logger.info(
        "Tool registry running in MCP mode transport=%s tools=%s",
        client.transport,
        sorted(MCP_PRIMARY_TOOL_NAMES),
    )
    return registry, client


__all__ = [
    "MCP_PRIMARY_TOOL_NAMES",
    "McpAgentTool",
    "ToolMode",
    "apply_mcp_adapters",
    "build_tool_registry",
]
