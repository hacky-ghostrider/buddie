# ADR-011: LangGraph Agent

- **Status:** Accepted  
- **Date:** 2026-08-07  
- **Sprint:** 11  
- **Depends on:** [ADR-008](ADR-008-RAG-Orchestration.md), [ADR-010.2](ADR-010.2-Agent-Evaluation-Preparation.md)

## Context

Sprints 9–10.2 delivered a complete evaluation architecture
(`EvaluationContext`, `ToolContract`, `ToolExecution`, `ToolTraceMapper`,
`ToolValidator`, DeepEval adapters, LangSmith tracing). Sprint 11 must add
the **first production LangGraph agent** without duplicating RAG logic or
refactoring evaluation.

## Decision

1. **One LangGraph agent** (`app/agent/`) with nodes: `planner` → `router` →
   `finalize`.
2. **RAG becomes tools**, not a parallel pipeline. `search_docs` and
   `summarize` reuse `RAGService` via `RAGToolBundle`.
3. **Explicit `AgentState`** carries question, planner output, contracts,
   executions, evaluation context, ids.
4. **Tools always return `ToolExecution`** with `ToolExecutionStatus` enum.
5. **Planner emits `ToolContract`s**; router executes; finalize builds
   `EvaluationContext`; `AgentService` validates and traces.
6. **No** memory, reflection, supervisor, swarm, multi-agent, HITL, MCP, or
   streaming in this sprint.

## Why RAG became a Tool

Agents choose capabilities dynamically. If RAG stays a hard-wired outer
pipeline, every new tool forces a fork. Wrapping `RAGService` preserves the
Sprint 8 use-case while letting the planner compose retrieve-then-summarize
(and future tools) under one evaluation contract.

## Why Planner exists

The planner is the architect: required / optional / alternative tools,
execution order, and contracts. Separating *decision* from *execution*
keeps policy reviewable and mockable without live LLMs.

## Why Router exists

The router is the foreman: run planned tools via a registry, collect
`ToolExecution`s, populate state. It never invents tool policy.

## Why AgentState

Explicit state is a flight recorder. Debugging “wrong tool / wrong order”
means reading typed fields — not reconstructing intent from chat strings.

## Why ToolExecution

Vendor-neutral observed invocation. LangGraph, LangSmith, and future MCP /
OpenAI Agents / CrewAI / AutoGen all map into the same record. Validators
stay vendor-blind.

## Why ToolExecutionStatus

String statuses drift (`"error"` vs `"failed"`). An enum gives a closed set
(`SUCCESS`, `FAILED`, `SKIPPED`, `TIMEOUT`, `RETRY`, `CANCELLED`) with
legacy coercion for Sprint 10.2 aliases.

## Why EvaluationContext

One aggregate DTO for metrics, tools, traces, and reports. The agent fills
it; DeepEval / ToolValidator / LangSmith consumers do not grow new kwargs.

## Why Tool Contracts

Contracts are the **spec** (args, order, bounds, latency, output type).
Goldens, planner output, and CI share one source of truth.

## Why ToolTraceMapper

Anti-corruption layer: LangSmith-like payloads → `ToolExecution` →
`ActualToolCall`. `ToolValidator` never parses LangSmith SDK objects. The
agent also produces `ToolExecution` directly; the mapper remains the bridge
when only traces are available.

## Architecture

```text
User
  ↓
LangGraph Agent (planner → router → finalize)
  ↓
RAG Tool (search_docs / summarize via RAGService)
Calculator Tool | Search Tool (mock)
  ↓
ToolExecution + ToolExecutionStatus
  ↓
EvaluationContext
  ↓
ToolValidator + DeepEval + LangSmith → Evaluation Report
```

## Consequences

- Evaluation architecture from Sprint 10.2 remains stable.
- Future frameworks plug in by mapping to `ToolExecution` / contracts.
- Sprint 12 can add MCP / multi-agent without rewriting validators.
