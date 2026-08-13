"""Buddie MCP client — discover and invoke tools via the official MCP SDK."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator, Literal
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp.context import build_call_meta
from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES

logger = logging.getLogger(__name__)

TransportKind = Literal["memory", "stdio", "http"]


class McpClientError(Exception):
    """Safe MCP client failure (connection, timeout, protocol)."""

    def __init__(self, message: str, *, code: str = "mcp_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync agent tool code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Already inside an event loop (e.g. pytest-asyncio) — use a side thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _parse_tool_payload(result: Any) -> dict[str, Any]:
    """Extract a structured dict from an MCP CallToolResult."""
    if getattr(result, "structuredContent", None):
        structured = result.structuredContent
        if isinstance(structured, dict):
            return structured

    contents = list(getattr(result, "content", None) or [])
    for block in contents:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"ok": not bool(getattr(result, "isError", False)), "raw": text}
        if isinstance(parsed, dict):
            return parsed
        return {"ok": True, "data": {"value": parsed}}

    if getattr(result, "isError", False):
        return {
            "ok": False,
            "error": "MCP tool returned an error without a structured body.",
            "error_code": "mcp_tool_error",
        }
    return {"ok": True, "data": {}}


def normalize_mcp_http_url(server_url: str) -> str:
    """Normalize a configured MCP Streamable HTTP URL.

    FastMCP serves Streamable HTTP at ``streamable_http_path`` (default
    ``/mcp``). If configuration provides only an origin, append that path.
    Does not invent hostnames — the URL host remains caller-configured.
    """
    raw = (server_url or "").strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    path = (parsed.path or "").rstrip("/")
    if path in {"", "/"}:
        return raw.rstrip("/") + "/mcp"
    return raw


class BuddieMcpClient:
    """MCP client adapter for Buddie.

    Supports:
    - ``memory``: in-process FastMCP (tests + same-process API mode)
    - ``stdio``: subprocess MCP server (local / container-friendly)
    - ``http``: Streamable HTTP URL from configuration (future remote server)

    Schemas are discovered via ``list_tools`` — not hardcoded on the client.
    """

    def __init__(
        self,
        *,
        transport: TransportKind = "memory",
        server: FastMCP | None = None,
        server_command: list[str] | None = None,
        server_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._transport = transport
        self._server = server
        self._server_command = list(server_command or [])
        raw_url = (server_url or "").strip() or None
        self._server_url = normalize_mcp_http_url(raw_url) if raw_url else None
        self._timeout = timedelta(seconds=max(0.1, float(timeout_seconds)))
        self._connected = False
        self._last_connect_error: str | None = None
        self._discovered_tools: list[dict[str, Any]] = []
        self._total_mcp_latency_ms = 0.0

        if transport == "memory" and server is None:
            raise McpClientError(
                "memory transport requires an in-process FastMCP server instance.",
                code="misconfigured",
            )
        if transport == "stdio" and not self._server_command:
            raise McpClientError(
                "stdio transport requires mcp_server_command configuration.",
                code="misconfigured",
            )
        if transport == "http" and not self._server_url:
            raise McpClientError(
                "http transport requires mcp_server_url configuration.",
                code="misconfigured",
            )
        if transport == "http" and self._server_url:
            parsed = urlparse(self._server_url)
            if parsed.scheme not in {"http", "https"}:
                raise McpClientError(
                    "mcp_server_url must be an http(s) URL.",
                    code="misconfigured",
                )

    @property
    def transport(self) -> str:
        return self._transport

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_connect_error(self) -> str | None:
        return self._last_connect_error

    @property
    def discovered_tools(self) -> list[dict[str, Any]]:
        return list(self._discovered_tools)

    @property
    def total_mcp_latency_ms(self) -> float:
        return self._total_mcp_latency_ms

    def connect(self) -> list[dict[str, Any]]:
        """Connect and discover tools. Safe to call repeatedly."""
        try:
            tools = _run_async(self._discover_async())
            self._connected = True
            self._last_connect_error = None
            self._discovered_tools = tools
            return tools
        except McpClientError as exc:
            self._connected = False
            self._last_connect_error = exc.message
            raise
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            message = "Unable to connect to the MCP server."
            self._last_connect_error = message
            logger.warning("MCP connect failed: %s", exc)
            raise McpClientError(message, code="connection_failed") from exc

    def list_tools(self) -> list[dict[str, Any]]:
        """Return discovered tool descriptors (connects if needed)."""
        if not self._discovered_tools:
            return self.connect()
        return list(self._discovered_tools)

    def ensure_expected_tools(
        self,
        expected: tuple[str, ...] | list[str] | None = None,
    ) -> list[str]:
        """Fail if an expected Buddie MCP tool is missing."""
        names = {tool["name"] for tool in self.list_tools()}
        required = list(expected or EXPECTED_MCP_TOOL_NAMES)
        missing = [name for name in required if name not in names]
        if missing:
            raise McpClientError(
                f"MCP server is missing expected tools: {', '.join(missing)}",
                code="tool_not_found",
            )
        return required

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        verified_employee_id: str | None = None,
        leave_request_confirmed: bool = False,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool and return a structured result envelope."""
        if self._discovered_tools:
            known = {tool["name"] for tool in self._discovered_tools}
            if name not in known:
                raise McpClientError(
                    f"MCP tool not found: {name}",
                    code="tool_not_found",
                )
        started = time.perf_counter()
        try:
            payload = _run_async(
                self._call_tool_async(
                    name,
                    dict(arguments or {}),
                    verified_employee_id=verified_employee_id,
                    leave_request_confirmed=leave_request_confirmed,
                    correlation_id=correlation_id,
                )
            )
            self._connected = True
            self._last_connect_error = None
            return payload
        except McpClientError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP call_tool failed: tool=%s error=%s", name, exc)
            message = str(exc).lower()
            if "not found" in message or "unknown tool" in message:
                raise McpClientError(
                    f"MCP tool not found: {name}",
                    code="tool_not_found",
                ) from exc
            raise McpClientError(
                "MCP tool invocation failed.",
                code="invocation_failed",
            ) from exc
        finally:
            self._total_mcp_latency_ms += (time.perf_counter() - started) * 1000.0

    def status_snapshot(self) -> dict[str, Any]:
        """Safe observability snapshot for Developer Mode."""
        return {
            "connected": self._connected,
            "transport": self._transport,
            "server_url": self._server_url,
            "tool_count": len(self._discovered_tools),
            "tools": [t.get("name") for t in self._discovered_tools],
            "last_error": self._last_connect_error,
            "mcp_latency_ms": round(self._total_mcp_latency_ms, 3),
        }

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        if self._transport == "memory":
            assert self._server is not None
            async with create_connected_server_and_client_session(
                self._server,
                read_timeout_seconds=self._timeout,
            ) as session:
                yield session
            return

        if self._transport == "stdio":
            command = self._server_command[0]
            args = list(self._server_command[1:])
            params = StdioServerParameters(command=command, args=args)
            async with stdio_client(params) as (read, write):
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=self._timeout,
                ) as session:
                    await session.initialize()
                    yield session
            return

        # Streamable HTTP — URL from configuration (no hardcoded host).
        import httpx
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        assert self._server_url is not None
        timeout = httpx.Timeout(
            self._timeout.total_seconds(),
            read=self._timeout.total_seconds(),
        )
        async with create_mcp_http_client(timeout=timeout) as http_client:
            async with streamable_http_client(
                self._server_url,
                http_client=http_client,
            ) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=self._timeout,
                ) as session:
                    await session.initialize()
                    yield session

    async def _discover_async(self) -> list[dict[str, Any]]:
        async with self._session() as session:
            listed = await session.list_tools()
            tools: list[dict[str, Any]] = []
            for tool in listed.tools:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema or {},
                        "output_schema": getattr(tool, "outputSchema", None),
                    }
                )
            return tools

    async def _call_tool_async(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        verified_employee_id: str | None,
        leave_request_confirmed: bool,
        correlation_id: str | None,
    ) -> dict[str, Any]:
        meta = build_call_meta(
            verified_employee_id=verified_employee_id,
            leave_request_confirmed=leave_request_confirmed,
            correlation_id=correlation_id,
        )
        async with self._session() as session:
            # Refresh discovery cache opportunistically.
            if not self._discovered_tools:
                listed = await session.list_tools()
                self._discovered_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema or {},
                        "output_schema": getattr(tool, "outputSchema", None),
                    }
                    for tool in listed.tools
                ]
            known = {tool["name"] for tool in self._discovered_tools}
            if known and name not in known:
                raise McpClientError(
                    f"MCP tool not found: {name}",
                    code="tool_not_found",
                )
            try:
                result = await session.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=self._timeout,
                    meta=meta or None,
                )
            except TimeoutError as exc:
                raise McpClientError(
                    f"MCP tool timed out: {name}",
                    code="timeout",
                ) from exc
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if "timeout" in message:
                    raise McpClientError(
                        f"MCP tool timed out: {name}",
                        code="timeout",
                    ) from exc
                raise

            payload = _parse_tool_payload(result)
            if getattr(result, "isError", False) and payload.get("ok") is not False:
                payload = {
                    "ok": False,
                    "tool": name,
                    "data": {},
                    "error": payload.get("error")
                    or payload.get("raw")
                    or "MCP tool reported an error.",
                    "error_code": payload.get("error_code") or "mcp_tool_error",
                }
            payload.setdefault("tool", name)
            return payload


__all__ = [
    "BuddieMcpClient",
    "McpClientError",
    "TransportKind",
    "normalize_mcp_http_url",
]
