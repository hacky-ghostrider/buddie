# Design Trade-offs (v1.0)

Frozen architecture decisions. Version 2 ideas belong in [ROADMAP.md](../../ROADMAP.md).

---

## Why LangGraph

**Choice:** LangGraph for the single-agent control flow (planner → router → finalize).

**Why:** Explicit graph state, typed transitions, and a clear seam for tool execution — better than an opaque agent loop for evaluation and debugging.

**Trade-off:** More boilerplate than a one-file ReAct script. Worth it because interviewers and production incidents need inspectable plans and tool histories.

**Not chosen:** CrewAI / AutoGen / multi-agent supervisors (v2).

---

## Why Chroma

**Choice:** Embedded Chroma with filesystem persistence (`CHROMA_PERSIST_DIRECTORY`).

**Why:** Zero ops for local/demo/CI, persistent volumes in Docker, enough for a focused RAG demo.

**Trade-off:** Not a managed distributed vector DB. Swap behind `VectorStore` ABC when scale demands Pinecone/Weaviate/pgvector.

---

## Why DeepEval

**Choice:** DeepEval for LLM-as-judge metrics via adapters.

**Why:** Strong faithfulness / hallucination / relevancy coverage; adapter keeps `EvaluationService` vendor-neutral.

**Trade-off:** Cost and non-determinism. Mitigated with `ENABLE_DEEPEVAL`, injectable measure functions, and offline demo/CI paths.

**Not chosen:** RAGAS as primary (v2 option behind the same `Metric` interface).

---

## Why LangSmith

**Choice:** LangSmith for execution traces behind `Tracer` / `NoOpTracer`.

**Why:** Best-in-class timeline for LangChain/LangGraph stacks; reports deep-link via `langsmith_run_url`.

**Trade-off:** Vendor SaaS dependency. Disabled by default; mapper isolates tool validation from LangSmith payloads.

---

## Why Tool Contracts

**Choice:** Declarative `ToolContract` + `ToolValidator` comparing expected vs actual tools/args/order.

**Why:** Agent correctness is behavioral, not only linguistic. Contracts are the agent analogue of API schemas / UI test expectations.

**Trade-off:** Authoring contracts for every golden. Pays off in regression suites and interview clarity.

---

## Why EvaluationContext

**Choice:** One enriched context object carrying question, retrieval, prompt, tools, answer, tokens, cost, LangSmith ids, correlation id.

**Why:** Prevents parallel “RAG eval” and “agent eval” stacks. Sprint 10.2 prepared the seam; Sprint 11 filled executions.

**Trade-off:** Fatter DTO. Better than N ad-hoc dicts across metrics and tracers.

---

## Why Planner

**Choice:** Dedicated planner node emitting required tools, order, and contracts (`PlannerDecision` in Sprint 12).

**Why:** Separates *policy* (what should run) from *mechanism* (running tools). Enables inspection before side effects; LLM planner can plug in later.

**Trade-off:** Rule-based planner is limited. Acceptable for v1; learnable planner is v2.

---

## Why Router

**Choice:** Tool router executes the plan against a registry (`search_docs`, `summarize`, …).

**Why:** Single place for invocation, latency, and `ToolExecution` / status recording. Graph stays stable when tools are added.

**Trade-off:** Less “autonomous” than free-form tool calling. Prefer determinism for evaluation.

---

## Why Quality Gates

**Choice:** Configurable PASS / WARNING / FAIL engine on top of evaluation reports + regression + benchmark history.

**Why:** `overall_score` alone is not a release policy. Gates encode SLOs (faithfulness floors, latency/cost ceilings, tool failures).

**Trade-off:** Threshold tuning. Thresholds live in Settings/env so ops can change policy without code changes.

---

## Why freeze at v1.0

Shipping Memory, MCP, multi-agent, and new metric vendors in the same release would blur the story. v1 proves **evaluation-grade architecture**; v2 extends capabilities on stable seams.
