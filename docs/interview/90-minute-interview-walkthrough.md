# 90-Minute Interview Walkthrough

Present this project like a staff engineer defending a production AI system — not like someone who followed a tutorial.

**Prep:** run `make demo` once before the interview so artifacts exist under `data/demo/`.

---

## 0:00–0:05 — Problem statement (5 min)

**Say:**

> Enterprises do not ship RAG because the demo looked good. They ship when retrieval, generation, agents, and evaluation are governed by contracts and release gates — the same way we gate microservices with tests and SLOs.

**Point to:** README overview + frozen v1.0 scope (no MCP/Memory/Multi-Agent yet).

**Close with:** “Version 1.0 is a Continuous AI Evaluation Platform around a production RAG + LangGraph agent.”

---

## 0:05–0:15 — High-level architecture (10 min)

**Draw / open:** [architecture/diagrams.md](../architecture/diagrams.md) component diagram.

**Talk track:**

1. Layered packages: ingestion → embeddings → vectorstore → retrieval → generation → orchestration → agent → evaluation → quality.
2. FastAPI is thin; `RAGService` owns the use-case.
3. Adapters (OpenAI, Chroma, DeepEval, LangSmith) sit behind ABCs.
4. Dependency rule: domain never imports vendor SDKs.

**Soundbite:** “If DeepEval types leak into orchestration, swapping vendors becomes a rewrite.”

---

## 0:15–0:25 — RAG pipeline (10 min)

**Walk:** PDF → chunk → embed → Chroma → retrieve Top-K → prompt templates → LLM → `RAGResponse` with latency breakdown.

**Emphasize:**

- Strategy for loaders/chunkers/embeddings/stores/LLMs
- Externalized prompts under `app/prompts/templates/`
- Latency fields as first-class observability

**Optional:** `GET /health`, `POST /api/v1/rag/query`

---

## 0:25–0:35 — LangGraph agent (10 min)

**Scenario:** `agent-tools-foundation-001` — “Summarize the leave policy from the employee handbook.”

**Flow:** Planner → contracts → Router → `search_docs` → `summarize` → finalize → `EvaluationContext`.

**Why it matters:** Correct *language* answers can still fail if the wrong tools ran. Agents are evaluated like UI flows: contract → trace → assert.

**Files:** `app/agent/planner.py`, `router.py`, `graph.py`, `service.py`

---

## 0:35–0:45 — DeepEval integration (10 min)

**Explain adapter boundary:**

`EvaluationContext` → `DeepEvalMetricAdapter` → `MetricResult` → `EvaluationReport`

**Metrics:** faithfulness, hallucination (inverted), answer relevancy, contextual precision/recall.

**CI story:** offline demo injects deterministic measure functions; live mode uses real DeepEval. Same report shape either way.

---

## 0:45–0:55 — LangSmith traces (10 min)

**Contrast:**

| | Traces | Metrics |
|--|--------|---------|
| Question | What happened? | How good? |
| Analogy | OpenTelemetry spans | JUnit + coverage |

Show `langsmith_run_url` on the report. Offline uses `NoOpTracer`; live needs `ENABLE_LANGSMITH=true`.

**Mapper story:** LangSmith payload → `ToolTraceMapper` → `ToolExecution` → validator (validator never parses LangSmith).

---

## 0:55–1:05 — Tool validation (10 min)

**Contract table:**

| Tool | Required args | Order |
|------|---------------|-------|
| `search_docs` | `query` | 0 |
| `summarize` | `document` | 1 |

**Assert:** expected tools == actual tools == `["search_docs", "summarize"]`, order preserved, `tool_validation.passed is True`.

**Interview line:** “We evaluate agents the way we evaluate UI flows.”

---

## 1:05–1:15 — Quality gates (10 min)

**Pipeline:** EvaluationReport → QualityGateEngine → PASS/WARNING/FAIL → BenchmarkHistory → `quality_report.*`

**Show:** thresholds from Settings (`MIN_FAITHFULNESS`, `PASS_THRESHOLD`, …), recommendations, CI workflow that fails on FAIL/regression.

**Enterprise framing:** release policy for AI, not vibes.

---

## 1:15–1:25 — Trade-offs and roadmap (10 min)

**Cover from** [design/tradeoffs.md](../design/tradeoffs.md):

- Why LangGraph, Chroma, DeepEval, LangSmith, tool contracts, EvaluationContext, planner, router, quality gates

**Roadmap honesty:** v2 = Memory, Reflection, MCP, RAGAS, Phoenix, OTel, multi-agent — deliberately **not** in v1.

---

## 1:25–1:30 — Demo (5 min)

```bash
make demo
```

Narrate the printed summary: agent tools → LangSmith ids → DeepEval scores → tool validation → quality decision → artifact paths.

If time: open `data/demo/reports/evaluation.html` and `data/demo/quality/demo_quality.html`.

---

## Closing line

> This is Version 1.0 of a production AI evaluation platform. The architecture is frozen on purpose — so we can talk about quality engineering, not feature sprawl.
