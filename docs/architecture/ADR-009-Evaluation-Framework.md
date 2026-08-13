# ADR-009: Tool-independent Evaluation Framework

- **Status:** Accepted  
- **Date:** 2026-08-07  
- **Sprint:** 9  

## Context

Sprints 1–8 delivered a production-style RAG pipeline ending in
`RAGResponse`. Teams still need a way to **score** answers and retrieval
quality without locking the codebase to any one vendor library
(DeepEval, RAGAS, LangSmith, Phoenix).

Gluing DeepEval calls into `RAGService` or FastAPI handlers would:

- couple business orchestration to a specific eval SDK
- make offline golden-set runs harder to test without network / LLM judges
- force a rewrite if the team later prefers RAGAS or an in-house metric

## Decision

1. Introduce `app/evaluation/` as a **tool-independent** evaluation layer.
2. Define `Metric` (ABC) with `evaluate()` / `name()` / `description()`.
3. Use `MetricRegistry` to register, enable/disable, and resolve metrics
   (plugin architecture).
4. Implement `EvaluationService` that accepts Question + optional expected
   answer + `RAGResponse` and returns a structured `EvaluationReport`.
5. Model `GoldenExample` as the golden-dataset contract (loading deferred).
6. Ship only **placeholder** metrics (`AnswerLengthMetric`,
   `ContextCountMetric`) to prove plumbing.
7. Configure via `ENABLE_EVALUATION`, `DEFAULT_PASS_THRESHOLD`,
   `METRIC_TIMEOUT`.
8. **Do not** integrate DeepEval, RAGAS, LangSmith, Phoenix, red teaming,
   dashboards, observability, or CI gates in this sprint.

## Why EvaluationService exists

It owns the **evaluation use-case**: run enabled metrics → aggregate →
report. Callers (CLI, future API, batch harness) share one entry point —
like a Spring `@Service` that does not know about DeepEval internals.

## Why Metric abstraction exists

Strategy pattern: every scorer looks the same to the service. Future
DeepEval / RAGAS adapters implement `Metric` and register themselves.
The Open/Closed Principle: add metrics without editing the orchestrator.

## Why Registry exists

Extensibility without edits to `EvaluationService`. Enable/disable metrics
per environment (cheap heuristics in CI, expensive LLM judges overnight).
Same idea as the document-loader extension registry (ADR-002).

## Why reports are structured

Free-text logs cannot be compared across runs, asserted in tests, or
aggregated for pass rates. `EvaluationReport` is a typed contract
(question, answer, documents, metrics, overall_score, passed, timestamps,
latency) suitable for JSON serialization and later dashboards.

## Why DeepEval is intentionally not integrated yet

Architecture before adapters. Integrating a vendor SDK first would bake
its types into the domain. Sprint 9 freezes the seams; Sprint 10+ can wrap
DeepEval behind `Metric` with minimal churn.

## Architecture

```text
Question
   ↓
RAGResponse
   ↓
EvaluationService
   ↓
Metric(s) via MetricRegistry
   ↓
EvaluationReport
```

## Consequences

- **Positive:** Eval layer is vendor-neutral and unit-testable with mocks.  
- **Positive:** Placeholder metrics prove registry + report aggregation.  
- **Positive:** GoldenExample ready for offline suites without loaders yet.  
- **Negative:** Placeholder scores are not semantic quality measures.  
- **Deferred:** DeepEval / RAGAS adapters, dataset loaders, CI gates,
  online feedback loops, dashboards.
