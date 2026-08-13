# Agent Tools Foundation — Interview Demonstration

**Scenario:** `agent-tools-foundation-001`  
**Dataset:** [`datasets/agent-tools-foundation-001.json`](../../datasets/agent-tools-foundation-001.json)  
**Sprint:** 10.2 (infrastructure) → **11 (LangGraph execution)**  

For the full Sprint 11 agent walkthrough see
[`langgraph-agent-demo.md`](langgraph-agent-demo.md).

---

## One-sentence pitch

> We evaluate agents the way we evaluate UI flows: declare a
> **contract**, capture a **trace**, map it to a **neutral model**, then
> **assert** — and Sprint 11 fills actual tool executions into the same seams.

---

## End-to-end pipeline

```text
Question
   ↓
Expected Tool Contract
   ↓
LangGraph Agent (Sprint 11)
   ↓
LangSmith Trace
   ↓
ToolTraceMapper
   ↓
Tool Validator
   ↓
DeepEval (via EvaluationContext)
   ↓
Evaluation Report
```

**Sprint 10.2 provided the infrastructure. Sprint 11 provides actual tool execution.**

---

## 1. Question

```text
Summarize the leave policy from the employee handbook.
```

Why this question: it *requires* a retrieve-then-summarize tool sequence.
A pure RAG answer can still be scored by DeepEval, but tool validation
checks whether an agent used the right tools in the right order.

---

## 2. Expected Tool Contract

| Tool | Required args | Order |
|------|---------------|-------|
| `search_docs` | `query` | 0 |
| `summarize` | `document` | 1 |

Expected arguments (subset match):

* `search_docs` → `{"query": "leave policy employee handbook"}`
* `summarize` → `{"document": "employee_handbook.pdf"}`

**File:** `app/evaluation/tool_validation/tool_contract.py`

---

## 3. EvaluationContext

One object carries everything evaluators need (question, retrieval, prompt,
tools, answer, model, latency, tokens, cost, LangSmith ids, correlation id).

**File:** `app/evaluation/context.py`

---

## 4. Tool Calls (Sprint 11)

| | Expected | Actual |
|--|----------|--------|
| Sprint 10.2 | `search_docs`, `summarize` | `[]` |
| Sprint 11 | `search_docs`, `summarize` | `search_docs`, `summarize` |

The scenario id stays `agent-tools-foundation-001` across both sprints.

```bash
pytest tests/test_agent.py::test_canonical_demo_agent_tools_foundation_001 -q
```

---

## 5. LangSmith Trace

LangSmith records *what happened*. DeepEval scores language/RAG quality; tool
validation scores agent contracts.

Dashboard for this scenario: **Planner → search_docs → summarize → Answer**.

> **To see real tool-validation signals in LangSmith, run `agent-tools-foundation-001`.**

---

## 6–8. Mapper → Validator → DeepEval → Report

Unchanged from Sprint 10.2: `ToolTraceMapper` → `ToolExecution` →
`ToolValidator`; DeepEval via `EvaluationContext`; structured
`EvaluationReport` with `tool_validation` + LangSmith URL.

See [`langgraph-agent-demo.md`](langgraph-agent-demo.md) for the full
production agent narrative.
