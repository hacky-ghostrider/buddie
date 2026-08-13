# ADR-010: AI Evaluation Automation Platform

- **Status:** Accepted  
- **Date:** 2026-08-07  
- **Sprint:** 10  

## Context

Sprint 9 delivered a vendor-neutral evaluation framework (`Metric`,
`MetricRegistry`, `EvaluationService`, `EvaluationReport`). Teams still
need production automation: real RAG quality metrics, execution traces,
tool-usage contracts for future agents, offline goldens, regression
comparison, and benchmark rollups — without baking DeepEval or LangSmith
types into orchestration code.

## Decision

1. **DeepEval Adapter** — implement DeepEval metrics as `Metric`
   Strategies (`DeepEvalMetricAdapter`) with explicit mapping to/from
   domain models. Invert hallucination so higher is always better.
2. **LangSmith Adapter** — introduce `Tracer` ABC + `LangSmithTracer` +
   `TracingService`. Persist run id / trace id / run URL on
   `EvaluationReport`. Use `NoOpTracer` when disabled.
3. **Tool Validation Framework** — domain models for expected vs actual
   tool calls (name, args, order, count, latency). No agent runtime in
   this sprint; Sprint 11 maps LangGraph / OpenAI Agents / CrewAI /
   AutoGen / MCP into `ActualToolCall`.
4. **Golden Dataset Loader** — versioned JSON with extended schema
   (tools, order, category, tags, difficulty).
5. **Evaluation Automation Service** — Load → RAG → Evaluate → Tools →
   Trace → multi-format Report.
6. **Regression Runner** — compare previous vs current JSON reports for
   score, latency, tool, and prompt/answer regressions.
7. **Benchmark Runner** — aggregate averages (faithfulness, hallucination,
   relevancy, context precision/recall, latency, tokens, cost, pass rate).

## Why DeepEval Adapter

DeepEval provides battle-tested LLM-as-judge metrics, but its
`LLMTestCase` / metric classes must not leak into `RAGService` or CI
orchestration. The adapter preserves Sprint 9’s Dependency Inversion:
swap DeepEval for RAGAS later by registering a different `Metric`.

## Why LangSmith Adapter

LangSmith answers “what happened?” (prompts, chunks, tokens, latency
trees). Evaluation answers “how good?”. Separating `Tracer` from
`Metric` prevents conflating observability with quality gates and allows
Phoenix / OpenTelemetry later.

## Why Tool Validation Framework

Agent quality is largely *tool contract* correctness. Vendor scorers do
not encode “must call `search_docs` before `summarize` with query X”.
Custom validation is the Selenium-style assertion layer for tool-using
agents — framework now, runtimes in Sprint 11.

## Why Regression Runner

Offline scores drift when prompts, models, or indexes change. Comparing
two report artifacts detects silent quality / latency / tool regressions
before release — analogous to comparing JUnit XML across builds.

## Why Benchmark Runner

Leadership and release gates need suite-level scorecards, not only
per-question rows. Benchmarks roll up averages and pass rate into a
single artifact under `BENCHMARK_DIRECTORY`.

## Architecture

```text
Golden Dataset
      │
      ▼
EvaluationAutomationService
      ├─ RAGRunner (RAGService or dry-run)
      ├─ EvaluationService (+ DeepEval adapters)
      ├─ ToolValidator
      ├─ TracingService (LangSmith / NoOp)
      └─ EvaluationReportWriter (JSON / CSV / HTML)
              │
              ├─► RegressionRunner (prev vs current)
              └─► BenchmarkRunner (suite averages)
```

## Consequences

- **Positive:** Vendor SDKs stay at the edges; domain reports stay stable.  
- **Positive:** Unit tests mock DeepEval / LangSmith / RAG — no live APIs.  
- **Positive:** Tool validation ready before agent implementation.  
- **Negative:** Live DeepEval still costs tokens and is non-deterministic;
  gate with `ENABLE_DEEPEVAL` and prefer mocked CI.  
- **Deferred:** LangGraph agents, MCP runtime, online feedback loops,
  dashboards, automated CI quality gates wiring.

## Related

- [ADR-009 — Evaluation Framework](ADR-009-Evaluation-Framework.md)
- [Sprint 10](../sprints/sprint-10-evaluation-automation.md)
