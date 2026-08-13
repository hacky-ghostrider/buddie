"""Run the Buddie MCP server as a standalone process.

Examples:
    python -m app.mcp
    python -m app.mcp --transport stdio
    python -m app.mcp --transport streamable-http --host 0.0.0.0 --port 8100

Configuration is environment-driven where possible (FASTMCP_HOST / PORT,
MCP_TRANSPORT, employee data path via Settings). No business logic lives here.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Literal

from app.config.settings import get_settings
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.mcp.server import run_buddie_mcp_server

logger = logging.getLogger(__name__)

TransportName = Literal["stdio", "streamable-http", "sse"]


def _resolve_transport(raw: str | None) -> TransportName:
    """Map client-facing MCP_TRANSPORT values to FastMCP run transports."""
    value = (raw or "stdio").strip().lower()
    if value in {"http", "streamable-http", "streamable_http"}:
        return "streamable-http"
    if value == "sse":
        return "sse"
    if value == "stdio":
        return "stdio"
    if value == "memory":
        raise SystemExit(
            "memory transport is in-process only; use streamable-http or stdio "
            "for python -m app.mcp"
        )
    raise SystemExit(f"Unsupported MCP transport: {raw}")


def main(argv: list[str] | None = None) -> None:
    env_transport = (
        os.environ.get("MCP_SERVER_TRANSPORT")
        or os.environ.get("MCP_TRANSPORT")
        or ""
    ).strip()
    parser = argparse.ArgumentParser(description="Buddie MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http", "sse", "http"),
        default=None,
        help="MCP transport (default: stdio, or MCP_TRANSPORT/MCP_SERVER_TRANSPORT)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host for HTTP transports (env FASTMCP_HOST / MCP_HOST otherwise)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port for HTTP transports (env FASTMCP_PORT / MCP_PORT otherwise)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    store = EmployeeStore(settings.employee_data_path)
    store.ensure_seeded()
    employee_service = EmployeeService(store=store)

    transport = _resolve_transport(args.transport or env_transport or "stdio")

    # Standalone process: employee tools only unless RAG is wired by the host.
    # Policy search still registers; without RAG it returns a safe error.
    run_buddie_mcp_server(
        transport=transport,
        host=args.host,
        port=args.port,
        employee_service=employee_service,
        rag_service=None,
    )


if __name__ == "__main__":
    main()
