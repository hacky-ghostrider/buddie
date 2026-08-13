# Sprint 11 — LangGraph Agent

**Status:** Complete  
**Depends on:** Sprint 8 (RAGService), Sprint 10.2 (evaluation hardening)

## Goal

Ship the first production-quality LangGraph agent. The RAG platform becomes
**tools** (`search_docs` / `summarize` via `RAGService`). Evaluation
architecture is reused — not rewritten.

## Delivered

| Item | Location |
|------|----------|
| Agent graph | `app/agent/graph.py` |
| Planner | `app/agent/planner.py` |
| Router | `app/agent/router.py` |
| AgentState | `app/agent/state.py` |
| RAG / calculator / search tools | `app/agent/tools/` |
| AgentService | `app/agent/service.py` |
| ToolExecutionStatus enum | `app/evaluation/tool_validation/tool_execution.py` |
| Canonical demo | `agent-tools-foundation-001` |
| Tests | `tests/test_agent.py` |
| ADR | [ADR-011](../architecture/ADR-011-LangGraph-Agent.md) |
| Interview demo | [langgraph-agent-demo.md](../demo/langgraph-agent-demo.md) |

## Out of scope (Sprint 12+)

Memory, reflection, self-correction, supervisor, swarm, multi-agent,
human-in-the-loop, MCP, streaming, CrewAI, AutoGen.

## Exit criteria

- [x] Production LangGraph agent  
- [x] Planner + Router  
- [x] RAG / Calculator / Search tools  
- [x] Tool Contracts + ToolExecutionStatus  
- [x] EvaluationContext integration  
- [x] LangSmith traces (via TracingService)  
- [x] Tool validation + DeepEval path  
- [x] Canonical demo PASS (`search_docs` → `summarize`)  
