# Sprint 16 — Docker + MCP HTTP Service Separation

**Status:** Complete  
**Depends on:** Sprint 15 (MCP) — frozen  
**Constraint:** Do not replace LangGraph / RuleBasedPlanner / Direct tools / memory transport

## 1. Sprint objective

Prove that Sprint 15 MCP works as a **separate HTTP service** across containers:

`buddie-ui` → `buddie-api` → MCP Streamable HTTP → `mcp-server` → existing services

No planner redesign. No new RAG. No Kubernetes.

## 2. Architecture before Sprint 16

```text
Planner → ToolRouter → ToolRegistry
                          ├── Direct AgentTool
                          └── McpAgentTool → BuddieMcpClient
                                              ↓
                                         memory (same process)
                                         or stdio
                                         or http (configured, not containerized)
                                              ↓
                                         FastMCP → EmployeeService / RAGService
```

## 3. Architecture after Sprint 16

```text
┌─────────────┐     ┌─────────────┐     MCP HTTP      ┌─────────────┐
│  buddie-ui  │────▶│ buddie-api  │──────────────────▶│ mcp-server  │
│  Streamlit  │     │ LangGraph   │  /mcp streamable  │ FastMCP     │
└─────────────┘     │ Planner     │                   └──────┬──────┘
                    │ ToolRouter  │                          │
                    └─────────────┘               ┌──────────┴──────────┐
                                                  ▼                     ▼
                                          EmployeeService          RAGService*
```

\*RAG in the standalone MCP container is optional; see limitations.

Direct mode and memory transport remain fully supported outside Docker.

## 4. Docker services

| Service | Role | Default port |
|---------|------|--------------|
| `mcp-server` | FastMCP Streamable HTTP | `8100` |
| `buddie-api` | FastAPI + LangGraph agent | `8000` |
| `buddie-ui` | Streamlit UI | `8501` |

Legacy single-container API: `docker compose --profile legacy up api`

## 5. Environment variables

| Variable | Values | Notes |
|----------|--------|-------|
| `BUDDIE_TOOL_MODE` | `direct` (default) \| `mcp` | Direct keeps Sprint 14 behavior |
| `MCP_TRANSPORT` | `memory` (default) \| `stdio` \| `http` | Client transport |
| `MCP_SERVER_URL` | URL | e.g. `http://mcp-server:8100/mcp` |
| `MCP_SERVER_COMMAND` | command string | stdio only |
| `MCP_TIMEOUT_SECONDS` | float | default `30` |
| `FASTMCP_HOST` / `FASTMCP_PORT` | bind | MCP server bind (Compose uses `0.0.0.0:8100`) |
| `MCP_ALLOWED_HOSTS` | CSV hosts | Optional DNS-rebinding allow-list |
| `API_BASE_URL` | URL | UI → API (`http://buddie-api:8000` in Compose) |

No deployment hostnames are hardcoded in Python business logic.

## 6. MCP HTTP endpoint

- Transport: official MCP **Streamable HTTP** (`streamable-http`)
- Path: FastMCP default **`/mcp`**
- Compose URL used by API: `http://mcp-server:8100/mcp`
- If `MCP_SERVER_URL` omits the path, the client appends `/mcp`

## 7. How to start the stack

```bash
docker compose up --build
```

Opens:

- UI: http://localhost:8501
- API: http://localhost:8000/health
- MCP: http://localhost:8100/mcp (MCP protocol, not a REST browser page)

## 8. How to run direct mode

**Local (no Docker):**

```bash
# default — no MCP env required
uv run uvicorn app.main:app --reload
uv run streamlit run frontend/app.py
```

**Docker (still default tool mode):**

```bash
# BUDDIE_TOOL_MODE defaults to direct; MCP server still runs but is unused
docker compose up --build
```

## 9. How to run MCP HTTP mode

**PowerShell:**

```powershell
$env:BUDDIE_TOOL_MODE="mcp"
$env:MCP_TRANSPORT="http"
$env:MCP_SERVER_URL="http://mcp-server:8100/mcp"
docker compose up --build
```

**bash:**

```bash
BUDDIE_TOOL_MODE=mcp \
MCP_TRANSPORT=http \
MCP_SERVER_URL=http://mcp-server:8100/mcp \
docker compose up --build
```

**Local MCP HTTP without Compose (API on host):**

```bash
# Terminal A — MCP server
python -m app.mcp --transport streamable-http --host 0.0.0.0 --port 8100

# Terminal B — API pointing at the MCP URL you configure
set BUDDIE_TOOL_MODE=mcp
set MCP_TRANSPORT=http
set MCP_SERVER_URL=http://<your-mcp-host>:8100/mcp
uv run uvicorn app.main:app --reload
```

Verify employee **E-1101**, run Sprint 14 multi-tool prompts, open Developer Mode:
`Protocol: MCP` and flow LangGraph → Planner → MCP → tools → Final Answer.

## 10. Test results

```bash
pytest -q
```

Sprint 16 adds focused HTTP/config/security/HITL/compose tests in
`tests/test_mcp_http_docker.py` (suite does **not** require a Docker daemon;
compose contract is YAML-parsed; optional `docker compose config` smoke skips
if Docker is unavailable).

Result: **433 passed, 1 skipped** (Sprint 15 baseline: **422 passed, 1 skipped**; +11 Sprint 16 tests).

## 11. Known limitations

1. **Standalone MCP container has no host RAG wiring** — `search_company_policy`
   registers but returns a safe error unless RAG is injected by the host process
   (same Sprint 15 standalone limitation). Memory/in-process MCP in the API still
   receives `RAGService`.
2. **Default Compose starts MCP even when tool mode is direct** — MCP is idle
   until `BUDDIE_TOOL_MODE=mcp` and `MCP_TRANSPORT=http`.
3. **Non-loopback binds disable FastMCP localhost DNS-rebinding by default** —
   set `MCP_ALLOWED_HOSTS` if you need an explicit Host allow-list.

## 12. Intentionally deferred (Sprint 17+)

- Shared RAG volume / sidecar wiring for MCP policy search
- AuthN/AuthZ between API and MCP
- Kubernetes / cloud deploy
- LLM planner replacement
- MCP federation / multi-server routing
