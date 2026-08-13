# Sprint 14 — Multi-Tool Agent

**Status:** Complete  
**Depends on:** Sprint 11 (LangGraph Agent), Agent Core polish (intent routing frozen)  
**Constraint:** Do not rewrite routing / RAG / UI; extend planner + tools only

## Goal

Evolve Buddie from a reliable single-tool / RAG assistant into a genuine
**multi-tool agent**: dynamic tool selection, state passing between tools,
combined answers, verification gates, human confirmation for writes, and
real Developer Mode execution metadata.

## Success state preserved (frozen)

Category-based intent routing, business-over-casual priority, greetings /
goodbye / thanks / acknowledgements, whitespace / numeric / gibberish /
punctuation handling, RAG routing, employee routing, and verification
enforcement remain intact. Prior robustness suite continues to pass.

## Target toolset (reused, not duplicated)

| Tool | Kind |
|------|------|
| `get_leave_balance` | Protected read |
| `get_leave_history` | Protected read |
| `get_employee_profile` | Protected read |
| `get_manager_information` | Protected read |
| `get_holiday_calendar` | Shared calendar |
| `check_leave_eligibility` | Protected check |
| `search_company_policy` | RAG policy lookup |
| `create_leave_request` | **Write** (HITL confirmation required) |

Additional existing tools (verify, holidays upcoming, payroll, attendance,
pending actions, `search_docs` / `summarize`, calculator, search) stay
registered and unchanged in behavior.

## Delivered

| Item | Location |
|------|----------|
| Employee multi-tool set | `app/agent/tools/employee_tools.py` |
| Structured employee service APIs | `app/employees/service.py`, `app/employees/models.py` |
| Multi-tool planner workflows | `app/agent/planner.py` |
| Combined answers + failure copy | `app/agent/router.py` |
| HITL pending leave + confirm | planner `pending_action` + frontend session metadata |
| `MULTI_TOOL` execution route metadata | `app/agent/service.py` |
| Evaluation metric placeholders | `app/evaluation/extension_points.py` |
| Multi-tool tests | `tests/test_multi_tool_agent.py` |

## Workflows

### A — Leave eligibility

```text
Verification → get_employee_profile → get_leave_balance
            → check_leave_eligibility
            → search_company_policy (only if needed, e.g. ≥10 days / "policy")
            → Final response
```

Example: *"Can I take 10 days of vacation?"*

### B — Carry-forward (hybrid)

```text
Verification → get_leave_balance + search_company_policy → Combined response
```

Response distinguishes employee balance vs policy text.  
Example: *"Can I carry forward my remaining vacation?"*

### C — Leave request (human-in-the-loop)

```text
Verification → profile → balance → eligibility
            → present result + ASK confirmation
            → ONLY after confirm → create_leave_request
```

`create_leave_request` never runs on intention alone. Cancel clears pending.

### D — Independent reads

```text
get_manager_information + get_holiday_calendar → Combined response
```

Example: *"Who is my manager and what holidays are coming up?"*  
Sequential execution (no new concurrency infrastructure).

## Architecture (unchanged spine)

```text
User question
    → classify_intent (frozen)
    → RuleBasedPlanner (multi-tool plans when needed)
    → ToolRouter (shared context: verified id, pending action)
    → Combined final answer
    → AgentService metadata (tools_invoked, MULTI_TOOL route, …)
```

- Tool B can use verified session state from Tool A’s context.
- Partial failures: friendly user copy; technical detail only in safe
  developer/evaluation metadata (no secrets / stack traces).

## Execution metadata (Developer / Evaluation)

Real fields only — never fabricated:

- `detected_intent` / `selected_route` (`MULTI_TOOL` when >1 non-RAG-pipeline tools)
- `verification_status` / `verified_employee_id`
- `tools_invoked[]` (name, status, arguments, result_summary, latency)
- `tool_execution_order`
- `awaiting_confirmation` / `pending_leave_request`
- `rag_used`, latency, LangSmith URL when enabled

Future evaluation contracts preserved as placeholders (RAG / Agent /
Safety / Performance metrics) without implementing the full engine.

## Out of scope (later)

MCP, Docker changes, GitHub Actions expansion, AWS / Azure / LocalStack,
complete Evaluation Engine, NL2SQL Learning Lab, Redis, UI redesign.

## Exit criteria

- [x] At least three genuine multi-tool workflows (A–D)
- [x] Dynamic tool selection (not “call all tools”)
- [x] State / verification between steps
- [x] Write-tool human confirmation
- [x] Real execution metadata in Developer Mode
- [x] Individual tool + workflow automated tests
- [x] Prior routing / robustness suites still green

## Test result

**403 passed**, 1 skipped (host Chroma smoke).

Key suite: `tests/test_multi_tool_agent.py`  
Regressions also covered in `tests/test_agent_routing.py`,
`tests/test_employee_data.py`, `tests/test_robustness_routing.py`.

## Interview relevance

Demo path:

1. Verify as `E-1101`
2. Eligibility ask → multi-tool trace
3. Carry-forward → balance + policy
4. “I want 5 days vacation next month” → confirm → write
5. Manager + holidays → combined answer
6. Developer Mode → `MULTI_TOOL` + ordered tool calls

## Limitations

- Rule-based planner (not LLM planner)
- Sequential tool execution only
- Draft leave dates proposed heuristically
- Full Evaluation Engine / MCP / NL2SQL still future
