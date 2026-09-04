# LangGraph Agent Demo

**Scenario:** `agent-tools-foundation-001`  
**Dataset:** [`datasets/agent-tools-foundation-001.json`](../../datasets/agent-tools-foundation-001.json)

Walkthrough for the LangGraph agent with tool contracts, tracing, and evaluation.

---

## One-sentence pitch

> The agent plans tools under contracts, executes them as `ToolExecution`
> records, fills `EvaluationContext`, and is scored by ToolValidator +
> DeepEval while LangSmith shows the timeline.

---

## End-to-end pipeline

```text
Question
   ↓
Planner (required tools + ToolContracts)
   ↓
Tool Router (search_docs → summarize)
   ↓
ToolExecution[]  +  ToolExecutionStatus
   ↓
LangSmith Trace (planner → tools → answer)
   ↓
EvaluationContext
   ↓
Tool Validation  +  DeepEval
   ↓
Evaluation Report
```

---

## 1. Question

```text
Summarize the leave policy from the employee handbook.
```

This question requires a retrieve-then-summarize sequence. A correct
*language* answer is not enough — tool validation checks **which** tools
ran and in **what order**.

---

## 2. Planner

The planner (rule-based by default; LLM-pluggable later) emits:

| Field | Value |
|-------|--------|
| Required tools | `search_docs`, `summarize` |
| Execution order | `search_docs` → `summarize` |
| Alternative tools | `search` (optional substitute family) |

**Why a planner node?** Separates *policy* (what should run) from
*mechanism* (running tools). Interviewers can inspect the plan before any
side effects.

---

## 3. Tool Contracts

| Tool | Required args | Order | Notes |
|------|---------------|-------|--------|
| `search_docs` | `query` | 0 | Reuses `RAGService` |
| `summarize` | `document` | 1 | Reuses cached RAG answer |

Expected arguments (subset match):

* `search_docs` → `{"query": "leave policy employee handbook"}`
* `summarize` → `{"document": "employee_handbook.pdf"}`

Contracts are the same Sprint 10.2 `ToolContract` objects — the agent does
not invent a second assertion language.

---

## 4. Tool Execution

```text
search_docs  →  ToolExecution(status=SUCCESS, …)
summarize    →  ToolExecution(status=SUCCESS, …)
```

| | Expected | Actual |
|--|----------|--------|
| Sprint 10.2 | `search_docs`, `summarize` | `[]` |
| Sprint 11 | `search_docs`, `summarize` | `search_docs`, `summarize` |

**PASS** when names, order, and arguments satisfy contracts.

`ToolExecutionStatus` is an enum (`SUCCESS`, `FAILED`, `SKIPPED`,
`TIMEOUT`, `RETRY`, `CANCELLED`) — no free-form status strings.

---

## 5. LangSmith Trace

Dashboard shape for this scenario:

```text
Planner
   ↓
search_docs
   ↓
summarize
   ↓
Answer
```

Captured fields include planner decision, selected tools, contracts,
arguments, order, latency, outputs, errors/retries, prompt/response when
present, final answer, plus `trace_id` / `run_id` / `run_url` /
`correlation_id`.

**How to inspect:** enable `ENABLE_LANGSMITH=true`, set
`LANGSMITH_API_KEY` + `LANGSMITH_PROJECT`, run the agent, open
`result.run_url` from the run result / evaluation report.

> Traces explain *what happened*. DeepEval scores *how good*. Tool
> validation scores *did we honour the contract*.

---

## 6. EvaluationContext

One object carries question, retrieved docs/chunks, prompt, tool calls /
results, answer, model, latency, tokens, cost, LangSmith ids, correlation
id, timestamp.

Metrics still read `context.answer` / `context.retrieved_documents`
(Sprint 9 compatibility). Agent fields ride along without signature
explosion.

---

## 7. Tool Validation

Reuses `ToolValidator` + `ToolTraceMapper`:

* Expected vs actual tool  
* Expected vs actual order  
* Expected vs actual arguments  
* Execution count / latency  
* Unexpected / missing / duplicate calls  
* Overall pass  

**Rule:** validators never parse LangSmith SDK objects. Mapper first.

---

## 8. DeepEval

DeepEval adapters consume `EvaluationContext` (faithfulness, relevancy,
etc. when enabled) and emit `MetricResult` rows. Vendor types stay inside
the adapter package.

---

## 9. Evaluation Report

Includes:

* Expected tools / actual tools / arguments / order / latency / pass  
* DeepEval scores (when enabled)  
* LangSmith URL  
* Optional `evaluation_context` snapshot  

---

## How this proves production-quality agent evaluation

1. **Behaviour + quality** — tool contracts and language metrics together.  
2. **Vendor isolation** — LangGraph / LangSmith stay behind mappers.  
3. **RAG reuse** — no duplicated retrieval/generation stack.  
4. **Deterministic demo** — same scenario id across Sprint 10.2 → 11.  
5. **Testable offline** — planner/router/tools/validation with mocked RAG
   and `NoOpTracer` (see `tests/test_agent.py`).

---

## Run (offline / mocked tests)

```bash
pytest tests/test_agent.py -q
pytest tests/test_agent.py::test_canonical_demo_agent_tools_foundation_001 -q
```

## Teaching appendix — files at a glance

| File | Role |
|------|------|
| `app/agent/graph.py` | LangGraph: planner → router → finalize |
| `app/agent/planner.py` | Structured plan + contracts |
| `app/agent/router.py` | Execute tools → `ToolExecution` |
| `app/agent/state.py` | Explicit `AgentState` |
| `app/agent/tools/rag_tool.py` | `search_docs` / `summarize` via `RAGService` |
| `app/agent/service.py` | Façade: run + validate + trace |
| `app/evaluation/context.py` | Evaluation aggregate |
| `app/evaluation/tool_validation/*` | Contracts, executions, mapper, validator |
