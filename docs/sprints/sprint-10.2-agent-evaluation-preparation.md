# Sprint 10.2 — Agent Evaluation Preparation (Architecture Hardening)

## Goal

Prepare the platform for Sprint 11 (LangGraph Agent) by hardening evaluation
architecture **without** adding agent runtime, MCP, tool calling, or new
end-user AI features.

## What we implemented

| Item | Location |
|------|----------|
| `EvaluationContext` aggregate | `app/evaluation/context.py` |
| `ToolContract` | `app/evaluation/tool_validation/tool_contract.py` |
| `ToolExecution` | `app/evaluation/tool_validation/tool_execution.py` |
| `ToolTraceMapper` | `app/evaluation/tool_validation/trace_mapper.py` |
| Canonical dataset | `datasets/agent-tools-foundation-001.json` |
| Scenario constants | `app/evaluation/scenarios.py` |
| Demo walkthrough | `docs/demo/agent-tools-foundation.md` |
| ADR | `docs/architecture/ADR-010.2-Agent-Evaluation-Preparation.md` |
| Tests | `tests/test_evaluation_foundation.py` |

## Explicitly out of scope

- LangGraph / agents / planner / router  
- MCP / tool calling runtime  
- CrewAI / AutoGen / reflection / memory  

## Canonical LangSmith reference scenario

**Name:** `agent-tools-foundation-001`

> To see real tool-validation signals in LangSmith, run `agent-tools-foundation-001`.

| | Expected tools | Actual tools |
|--|----------------|--------------|
| Sprint 10.2 | `search_docs`, `summarize` | `[]` (no agent yet) |
| Sprint 11 | `search_docs`, `summarize` | `search_docs`, `summarize` |

Scenario expectations stay unchanged in Sprint 11.

## Configuration

No new settings required. Existing LangSmith / tool-validation / golden-path
flags from Sprint 10 continue to apply.

## Exit criteria

- Evaluation / tracing / tool validation boundaries clear  
- Future agent integration plugs into `ToolExecution` + `EvaluationContext`  
- Sprint 11 can implement LangGraph without further eval architecture refactors  
- Mocked pytest coverage; no live API calls  

## Related

- [ADR-010.2](../architecture/ADR-010.2-Agent-Evaluation-Preparation.md)  
- [Demo walkthrough](../demo/agent-tools-foundation.md)  
- [Sprint 10](sprint-10-evaluation-automation.md)
