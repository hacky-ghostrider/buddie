"""Buddie MCP server — official FastMCP over existing services."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from app.employees.service import EmployeeService
from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES, TOOL_DESCRIPTIONS
from app.mcp.tools import BuddieMcpToolBridge
from app.orchestration.rag_service import RAGService

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _transport_security_for_bind_host(
    host: str | None,
) -> TransportSecuritySettings | None:
    """Configure DNS-rebinding rules for the chosen bind host.

    FastMCP auto-enables localhost-only Host checks when the constructor
    default loopback host is used. Container binds (e.g. ``0.0.0.0``) must
    not keep that localhost-only allow-list — otherwise Docker service-name
    Host headers are rejected.

    Optional ``MCP_ALLOWED_HOSTS`` (comma-separated, supports ``host:*``)
    re-enables protection with an explicit allow-list. No hostnames are
    hardcoded in business logic.
    """
    if host is None or host in _LOOPBACK_HOSTS:
        # Let FastMCP apply its default localhost protection.
        return None

    allowed_raw = (os.environ.get("MCP_ALLOWED_HOSTS") or "").strip()
    if allowed_raw:
        allowed_hosts = [part.strip() for part in allowed_raw.split(",") if part.strip()]
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_buddie_mcp_server(
    *,
    employee_service: EmployeeService | None = None,
    rag_service: RAGService | None = None,
    name: str = "buddie-mcp",
    host: str | None = None,
    port: int | None = None,
) -> FastMCP:
    """Create a FastMCP server exposing Buddie's approved toolset.

    Tools delegate to ``EmployeeService`` / ``RAGService`` — no duplicated
    business logic. Protected tools read verified employee id from MCP request
    meta (set by the Buddie MCP client), never from untrusted user ids.

    ``host`` / ``port`` are applied at construction time so FastMCP transport
    security matches the bind address (required for container HTTP).
    """
    employees = employee_service or EmployeeService()
    employees.ensure_ready()
    bridge = BuddieMcpToolBridge(
        employee_service=employees,
        rag_service=rag_service,
    )
    server_kwargs: dict[str, Any] = {"json_response": True}
    if host is not None:
        server_kwargs["host"] = host
    if port is not None:
        server_kwargs["port"] = port
    security = _transport_security_for_bind_host(host)
    if security is not None:
        server_kwargs["transport_security"] = security
    server = FastMCP(name, **server_kwargs)

    @server.tool(
        name="get_leave_balance",
        description=TOOL_DESCRIPTIONS["get_leave_balance"],
    )
    def get_leave_balance(ctx: Context) -> dict[str, Any]:
        return bridge.get_leave_balance(ctx)

    @server.tool(
        name="get_leave_history",
        description=TOOL_DESCRIPTIONS["get_leave_history"],
    )
    def get_leave_history(
        year: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return bridge.get_leave_history(year=year, ctx=ctx)

    @server.tool(
        name="get_employee_profile",
        description=TOOL_DESCRIPTIONS["get_employee_profile"],
    )
    def get_employee_profile(ctx: Context) -> dict[str, Any]:
        return bridge.get_employee_profile(ctx)

    @server.tool(
        name="get_manager_information",
        description=TOOL_DESCRIPTIONS["get_manager_information"],
    )
    def get_manager_information(ctx: Context) -> dict[str, Any]:
        return bridge.get_manager_information(ctx)

    @server.tool(
        name="get_holiday_calendar",
        description=TOOL_DESCRIPTIONS["get_holiday_calendar"],
    )
    def get_holiday_calendar(
        country: str | None = None,
        year: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return bridge.get_holiday_calendar(country=country, year=year, ctx=ctx)

    @server.tool(
        name="check_leave_eligibility",
        description=TOOL_DESCRIPTIONS["check_leave_eligibility"],
    )
    def check_leave_eligibility(
        leave_type: str = "VACATION",
        requested_days: float = 1,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return bridge.check_leave_eligibility(
            leave_type=leave_type,
            requested_days=requested_days,
            ctx=ctx,
        )

    @server.tool(
        name="search_company_policy",
        description=TOOL_DESCRIPTIONS["search_company_policy"],
    )
    def search_company_policy(
        query: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return bridge.search_company_policy(query=query, ctx=ctx)

    @server.tool(
        name="create_leave_request",
        description=TOOL_DESCRIPTIONS["create_leave_request"],
    )
    def create_leave_request(
        start_date: str,
        end_date: str,
        leave_type: str = "VACATION",
        reason: str = "Employee leave request",
        confirmed: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        return bridge.create_leave_request(
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            reason=reason,
            confirmed=confirmed,
            ctx=ctx,
        )

    logger.info(
        "Buddie MCP server built: tools=%s",
        list(EXPECTED_MCP_TOOL_NAMES),
    )
    return server


def run_buddie_mcp_server(
    *,
    transport: Literal["stdio", "streamable-http", "sse"] = "stdio",
    host: str | None = None,
    port: int | None = None,
    employee_service: EmployeeService | None = None,
    rag_service: RAGService | None = None,
) -> None:
    """Run the Buddie MCP server (stdio local; streamable-http for containers).

    Host/port come from arguments or ``FASTMCP_HOST`` / ``FASTMCP_PORT`` /
    ``MCP_HOST`` / ``MCP_PORT`` — no hardcoded deployment hostnames.
    """
    bind_host = host
    if bind_host is None:
        bind_host = (
            (os.environ.get("FASTMCP_HOST") or os.environ.get("MCP_HOST") or "").strip()
            or None
        )
    bind_port = port
    if bind_port is None:
        raw_port = (os.environ.get("FASTMCP_PORT") or os.environ.get("MCP_PORT") or "").strip()
        if raw_port:
            bind_port = int(raw_port)

    server = build_buddie_mcp_server(
        employee_service=employee_service,
        rag_service=rag_service,
        host=bind_host,
        port=bind_port,
    )
    logger.info(
        "Starting Buddie MCP server transport=%s host=%s port=%s path=%s",
        transport,
        server.settings.host,
        server.settings.port,
        server.settings.streamable_http_path,
    )
    server.run(transport=transport)


__all__ = [
    "EXPECTED_MCP_TOOL_NAMES",
    "build_buddie_mcp_server",
    "run_buddie_mcp_server",
]
