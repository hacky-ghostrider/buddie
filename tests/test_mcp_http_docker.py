"""Sprint 16 — MCP HTTP transport, config, security, and compose contracts."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from app.config.settings import Settings, get_settings
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.mcp.adapter import build_tool_registry
from app.mcp.client import BuddieMcpClient, McpClientError, normalize_mcp_http_url
from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES
from app.mcp.server import build_buddie_mcp_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def employee_service(tmp_path: Path) -> EmployeeService:
    store = EmployeeStore(tmp_path / "employees.json")
    store.seed()
    return EmployeeService(store)


@pytest.fixture
def mock_rag_service() -> MagicMock:
    service = MagicMock()
    service.query.side_effect = lambda request: MagicMock(
        answer="policy excerpt",
        retrieved_documents=[],
    )
    return service


@pytest.fixture
def http_mcp_server_url(employee_service: EmployeeService) -> str:
    """Start FastMCP Streamable HTTP on an ephemeral loopback port."""
    port = _free_port()
    server = build_buddie_mcp_server(
        employee_service=employee_service,
        rag_service=None,
        host="127.0.0.1",
        port=port,
    )
    thread = threading.Thread(
        target=lambda: server.run(transport="streamable-http"),
        daemon=True,
    )
    thread.start()

    deadline = time.time() + 8.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        pytest.fail("MCP HTTP server did not start in time")

    return f"http://127.0.0.1:{port}/mcp"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_direct_mode_remains_default() -> None:
    settings = Settings()
    assert settings.buddie_tool_mode == "direct"
    assert settings.mcp_transport == "memory"


def test_http_transport_and_server_url_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("BUDDIE_TOOL_MODE", "mcp")
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    monkeypatch.setenv("MCP_SERVER_URL", "http://mcp-server:8100/mcp")
    monkeypatch.setenv("MCP_TIMEOUT_SECONDS", "15")
    settings = Settings()
    assert settings.buddie_tool_mode == "mcp"
    assert settings.mcp_transport == "http"
    assert settings.mcp_server_url == "http://mcp-server:8100/mcp"
    assert settings.mcp_timeout_seconds == 15.0
    get_settings.cache_clear()


def test_normalize_mcp_http_url_appends_default_path() -> None:
    assert normalize_mcp_http_url("http://mcp-server:8100") == "http://mcp-server:8100/mcp"
    assert (
        normalize_mcp_http_url("http://mcp-server:8100/mcp")
        == "http://mcp-server:8100/mcp"
    )


def test_http_client_requires_server_url() -> None:
    with pytest.raises(McpClientError) as exc:
        BuddieMcpClient(transport="http", server_url="")
    assert exc.value.code == "misconfigured"


# ---------------------------------------------------------------------------
# MCP HTTP — discover / invoke / structured result
# ---------------------------------------------------------------------------


def test_mcp_http_discover_and_invoke(http_mcp_server_url: str) -> None:
    client = BuddieMcpClient(
        transport="http",
        server_url=http_mcp_server_url,
        timeout_seconds=20.0,
    )
    tools = client.connect()
    names = {tool["name"] for tool in tools}
    assert set(EXPECTED_MCP_TOOL_NAMES) <= names

    payload = client.call_tool(
        "get_leave_balance",
        {},
        verified_employee_id="E-1101",
    )
    assert payload["ok"] is True
    assert payload["tool"] == "get_leave_balance"
    assert "leave_balance" in payload["data"]
    assert payload["data"]["employee_id"] == "E-1101"


def test_mcp_http_origin_url_without_path_works(http_mcp_server_url: str) -> None:
    origin = http_mcp_server_url.rsplit("/mcp", 1)[0]
    client = BuddieMcpClient(transport="http", server_url=origin, timeout_seconds=20.0)
    client.connect()
    payload = client.call_tool(
        "get_employee_profile",
        {},
        verified_employee_id="E-1101",
    )
    assert payload["ok"] is True


# ---------------------------------------------------------------------------
# Security / HITL over HTTP
# ---------------------------------------------------------------------------


def test_mcp_http_trusted_metadata_and_spoof_ignored(http_mcp_server_url: str) -> None:
    client = BuddieMcpClient(
        transport="http",
        server_url=http_mcp_server_url,
        timeout_seconds=20.0,
    )
    client.connect()

    denied = client.call_tool(
        "get_leave_balance",
        {"employee_id": "E-1102"},
    )
    assert denied["ok"] is False
    assert denied.get("error_code") == "not_verified"

    allowed = client.call_tool(
        "get_leave_balance",
        {"employee_id": "E-1102"},
        verified_employee_id="E-1101",
    )
    assert allowed["ok"] is True
    assert allowed["data"]["employee_id"] == "E-1101"


def test_mcp_http_create_leave_requires_hitl(http_mcp_server_url: str) -> None:
    client = BuddieMcpClient(
        transport="http",
        server_url=http_mcp_server_url,
        timeout_seconds=20.0,
    )
    client.connect()

    denied = client.call_tool(
        "create_leave_request",
        {
            "leave_type": "VACATION",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "confirmed": False,
        },
        verified_employee_id="E-1101",
    )
    assert denied["ok"] is False
    assert denied.get("error_code") == "confirmation_required"

    allowed = client.call_tool(
        "create_leave_request",
        {
            "leave_type": "VACATION",
            "start_date": "2026-10-01",
            "end_date": "2026-10-03",
            "reason": "Sprint 16 HTTP HITL",
            "confirmed": True,
        },
        verified_employee_id="E-1101",
        leave_request_confirmed=True,
    )
    assert allowed["ok"] is True
    assert allowed["data"].get("created") is True


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_mcp_http_unavailable_falls_back_to_direct(
    mock_rag_service: MagicMock,
    employee_service: EmployeeService,
) -> None:
    registry, client = build_tool_registry(
        mock_rag_service,
        employee_service=employee_service,
        tool_mode="mcp",
        mcp_transport="http",
        mcp_server_url="http://127.0.0.1:59999/mcp",
        mcp_timeout_seconds=1.0,
    )
    assert client is None or not getattr(client, "connected", False)
    assert registry.has("get_leave_balance")
    tool = registry.get("get_leave_balance")
    assert tool.name == "get_leave_balance"


# ---------------------------------------------------------------------------
# Docker compose contract (no Docker daemon required)
# ---------------------------------------------------------------------------


def test_docker_compose_defines_sprint16_services() -> None:
    compose_path = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = document["services"]
    assert "buddie-api" in services
    assert "buddie-ui" in services
    assert "mcp-server" in services

    mcp_env = services["mcp-server"]["environment"]
    assert mcp_env["FASTMCP_PORT"] == "8100"
    api_env = services["buddie-api"]["environment"]
    assert api_env["BUDDIE_TOOL_MODE"] == "${BUDDIE_TOOL_MODE:-direct}"
    assert api_env["MCP_TRANSPORT"] == "${MCP_TRANSPORT:-memory}"
    assert "mcp-server:8100" in api_env["MCP_SERVER_URL"]
    assert services["buddie-ui"]["environment"]["API_BASE_URL"].startswith(
        "${API_BASE_URL:-http://buddie-api"
    )

    command = " ".join(services["mcp-server"]["command"])
    assert "streamable-http" in command
    assert "0.0.0.0" in command


def test_docker_available_optional_smoke() -> None:
    """Skip unless Docker is present — does not make the suite Docker-dependent."""
    docker = pytest.importorskip("subprocess")
    result = docker.run(
        ["docker", "compose", "config", "--services"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose unavailable: {result.stderr.strip()}")
    names = set(result.stdout.split())
    assert {"buddie-api", "buddie-ui", "mcp-server"} <= names
