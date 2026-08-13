# Sprint 9 — Evaluation Framework

## Goal

Design a **tool-independent** evaluation architecture that scores a
`RAGResponse` and produces a structured `EvaluationReport`.

No DeepEval, RAGAS, LangSmith, Phoenix, red teaming, dashboards,
observability, or CI/CD gates in this sprint — only the seams those tools
will plug into later.

```text
Question
   ↓
RAGResponse
   ↓
EvaluationService
   ↓
Metrics (via MetricRegistry)
   ↓
EvaluationReport
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | What is AI evaluation? | Measure answer + retrieval quality, not just HTTP 200 |
| 2 | LLMs ≠ traditional tests | Graded behavior, not exact `assertEquals` |
| 3 | Accuracy alone fails | Need faithfulness, relevance, groundedness, latency, … |
| 4 | Retrieval vs generation eval | Right books? vs good book report? |
| 5 | Offline vs online | Golden regression suite vs live feedback |
| 6 | Human vs automated | Humans catch nuance; automation scales |
| 7 | Tool-independent framework | Own `Metric` / `Report`; wrap vendors later |
| 8 | Golden Dataset | Curated Q → expected answer/sources |
| 9 | Good eval data | Representative, labeled, versioned, failure-mode heavy |
| 10 | Determinism | Same inputs → same scores (avoid flaky CI) |
| 11 | Registry / plugins | Add metrics without editing the service |
| 12 | Interviews | Faithfulness vs relevance; why abstract DeepEval? |

### Analogies

- Evaluation ≈ a QA regression suite for answers that aren’t binary pass/fail.
- Retrieval eval ≈ did the librarian fetch the right books?
- Generation eval ≈ is the book report good *and* faithful to those books?
- Golden dataset ≈ curated exam answer key (offline).
- Online eval ≈ production thumbs-down / escalation rate.
- `Metric` ABC ≈ Java interface; DeepEval later is one implementation.
- `MetricRegistry` ≈ Spring bean registry / plugin catalog.
- Structured report ≈ JUnit XML — machine-comparable, not a free-text log.

## What we tried to achieve

- `Metric` ABC (`evaluate`, `name`, `description`)
- `MetricRegistry` (register, enable/disable, resolve by name)
- `EvaluationService` (Question + optional expected answer + `RAGResponse` → report)
- `EvaluationReport` (metrics, overall_score, passed, latency, timestamp)
- `GoldenExample` model (dataset **loading deferred**)
- Placeholder metrics only: `AnswerLengthMetric`, `ContextCountMetric`
- Settings: `ENABLE_EVALUATION`, `DEFAULT_PASS_THRESHOLD`, `METRIC_TIMEOUT`
- Domain exceptions + centralized logging (no `print`)
- Pytest with mock / placeholder metrics (no DeepEval / RAGAS)
- Explicitly **no** vendor eval SDKs, dashboards, or CI gates

## Architecture

```text
RAGResponse (from Sprint 8)
      │
      ▼
EvaluationService                 app/evaluation/
      │
      ├─► MetricRegistry
      │        │
      │        ├─ AnswerLengthMetric   (placeholder)
      │        └─ ContextCountMetric   (placeholder)
      │
      └─► MetricResult[]
      │
      ▼
EvaluationReport {
  question, answer, retrieved_documents,
  metrics, overall_score, passed,
  evaluation_time, latency
}
```

### Folder responsibilities

| Path | Responsibility |
|------|----------------|
| `metrics/base.py` | `Metric` ABC |
| `metrics/answer_length.py` | Placeholder length scorer |
| `metrics/context_count.py` | Placeholder retrieved-count scorer |
| `registry.py` | Plugin catalog (enable/disable by name) |
| `evaluator.py` | `EvaluationService` use-case |
| `report.py` | `EvaluationReport` contract + `build()` |
| `models.py` | `EvaluationContext`, `MetricResult`, `GoldenExample` |
| `exceptions.py` | Domain errors |
| `base.py` | Stable re-exports |

### Why reports are structured

Free-text logs cannot be compared across runs, asserted in CI, or aggregated
for pass rates. A typed report is the evaluation equivalent of a test result
artifact.

### Why DeepEval is not integrated yet

Architecture before adapters. Integrating a vendor SDK first would bake its
types into the domain. Sprint 9 freezes the seams; later sprints wrap DeepEval
/ RAGAS behind `Metric` with minimal churn.

## Configuration

| Variable | Default | Validation |
|----------|---------|------------|
| `ENABLE_EVALUATION` | `true` | Boolean master switch |
| `DEFAULT_PASS_THRESHOLD` | `0.7` | Must be in `[0, 1]` |
| `METRIC_TIMEOUT` | `30` | Must be `> 0` (seconds, per metric) |

## Explicitly out of scope

DeepEval, RAGAS, LangSmith, Phoenix, AI red teaming, dashboards,
observability platforms, CI/CD quality gates, golden dataset file loaders,
weighted metric aggregation, LLM-as-judge.

## Exit criteria

- Evaluation package independent of DeepEval / RAGAS / LangSmith / Phoenix  
- `Metric` ABC implemented by every scorer  
- `MetricRegistry` supports register / enable / disable / get  
- `EvaluationService` aggregates metrics into `EvaluationReport`  
- Placeholder metrics only (length + context count)  
- `GoldenExample` model exists; **no** dataset loading yet  
- Settings validated; logging via centralized logger (no `print`)  
- Unit tests cover registration, execution, empty registry, failures, scores  
- ADR-009 recorded  

## Interview soundbite

> "I built a tool-independent evaluation layer: `EvaluationService` runs
> pluggable `Metric` strategies from a registry and emits a structured
> `EvaluationReport`. DeepEval or RAGAS can implement the same `Metric`
> interface later without rewriting the RAG orchestrator — same idea as
> keeping Selenium page objects behind interfaces so the runner stays stable."

## Related ADR

[ADR-009 — Evaluation Framework](../architecture/ADR-009-Evaluation-Framework.md)
