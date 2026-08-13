"""Buddie MCP interoperability layer (Sprint 15).

Exposes existing employee / policy tools through an official MCP server and
lets Buddie consume them via an MCP client adapter — without replacing
LangGraph, the planner, or direct ToolRegistry execution.
"""

from app.mcp.adapter import (
    MCP_PRIMARY_TOOL_NAMES,
    apply_mcp_adapters,
    build_tool_registry,
)
from app.mcp.client import BuddieMcpClient, McpClientError
from app.mcp.server import (
    EXPECTED_MCP_TOOL_NAMES,
    build_buddie_mcp_server,
    run_buddie_mcp_server,
)

__all__ = [
    "MCP_PRIMARY_TOOL_NAMES",
    "EXPECTED_MCP_TOOL_NAMES",
    "BuddieMcpClient",
    "McpClientError",
    "apply_mcp_adapters",
    "build_buddie_mcp_server",
    "build_tool_registry",
    "run_buddie_mcp_server",
]
