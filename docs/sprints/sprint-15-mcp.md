# Sprint 15 — MCP Tool Integration

**Status:** Complete  
**Depends on:** Sprint 14 (multi-tool agent) — frozen  
**Constraint:** Do not replace LangGraph / RuleBasedPlanner / RAG; MCP is an interoperability layer

## Goal

Expose Buddie's approved tools through an official MCP server and consume them
via an MCP client adapter, while preserving Direct tool mode as fallback.

## Tool mode

| Mode | Setting | Behavior |
|------|---------|----------|
| Direct (default) | `BUDDIE_TOOL_MODE=direct` | Existing `ToolRegistry` tools |
| MCP | `BUDDIE_TOOL_MODE=mcp` | Primary tools execute via MCP |

Transport (`MCP_TRANSPORT`): `memory` (default, in-process), `stdio`, `http`.

## MCP tools

`get_leave_balance`, `get_leave_history`, `get_employee_profile`,
`get_manager_information`, `get_holiday_calendar`, `check_leave_eligibility`,
`search_company_policy`, `create_leave_request`

## Layout

```text
app/mcp/
  server.py    FastMCP server
  tools.py     Bridge → EmployeeService / RAGService
  schemas.py   Descriptions + contracts
  client.py    Discover + invoke
  adapter.py   AgentTool MCP adapters + Direct/MCP boundary
  context.py   Trusted verified employee meta
```

## Demo

```bash
set BUDDIE_TOOL_MODE=mcp
set MCP_TRANSPORT=memory
```

Then verify E-1101 and run Sprint 14 multi-tool prompts; Developer Mode shows
`Protocol: MCP` and the LangGraph → Planner → MCP → tools flow.
