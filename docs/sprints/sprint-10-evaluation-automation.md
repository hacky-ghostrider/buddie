# Sprint 10 — AI Evaluation Automation Platform

## Goal

Transform the RAG platform into a **production-grade AI Evaluation
Automation Platform** that evaluates:

1. RAG quality (DeepEval adapters)
2. LLM quality (same metric suite + token/cost fields)
3. Agent tool usage (**framework only** — no LangGraph yet)
4. Regression over time
5. Execution traces (LangSmith adapter)

without tightly coupling business logic to any vendor.

```text
                    RAG Pipeline
                         │
                         ▼
                 Evaluation Service
                         │
      ┌──────────────────┼────────────────────┐
      ▼                  ▼                    ▼
 DeepEval Adapter   LangSmith Adapter   Tool Validation
      │                  │                    │
      └──────────────────┼────────────────────┘
                         ▼
                  Evaluation Report
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | Monitoring vs Observability vs Evaluation vs Testing | Metrics/alerts vs “why” traces vs quality scores vs assertions |
| 2 | DeepEval | LLM-as-judge / RAG quality metrics behind adapters |
| 3 | LangSmith | Execution traces, prompts, tokens, run URLs — not score gates |
| 4 | Traces ≠ metrics | What happened vs how good |
| 5 | Tool validation | Custom contract asserts; vendors don’t own your tool policy |
| 6 | Offline evaluation | Golden datasets, repeatable, CI-friendly |
| 7 | Regression | Compare previous vs current runs |
| 8 | Golden datasets | Curated Q → answer/sources/tools |
| 9 | Agent evaluation | Tool order/args/count foundation for Sprint 11 |
| 10 | Interviews | Adapter pattern, faithfulness vs relevancy, why invert hallucination |

## What we implemented

- `app/evaluation/deepeval/` — adapter, metrics catalog, mapping
- `app/tracing/` — `Tracer` ABC, LangSmith adapter, `TracingService`
- `app/evaluation/tool_validation/` — expectations, comparator, validator, report
- `datasets/golden_dataset.json` — extended schema
- `scripts/run_evaluation.py` — single + batch automation
- `scripts/run_regression.py` — score / latency / tool / prompt diffs
- `scripts/run_benchmark.py` — suite averages + pass rate
- JSON / CSV / HTML report writers
- Settings: LangSmith, DeepEval, tool validation, report/benchmark dirs
- Pytest with mocked OpenAI / DeepEval / LangSmith (no live API calls)
- ADR-010

## Explicitly out of scope

- LangGraph agent implementation
- MCP tool runtime
- Agent tool calling in production code
- Live DeepEval / LangSmith calls in unit tests

Those belong to **Sprint 11**.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `LANGSMITH_API_KEY` | `""` | LangSmith API key |
| `LANGSMITH_PROJECT` | `rag-evaluation` | Project name |
| `ENABLE_LANGSMITH` | `false` | Trace recording switch |
| `ENABLE_DEEPEVAL` | `true` | Register DeepEval adapters |
| `ENABLE_TOOL_VALIDATION` | `true` | Run tool validator in automation |
| `REPORT_DIRECTORY` | `./data/reports` | JSON/CSV/HTML output |
| `BENCHMARK_DIRECTORY` | `./data/benchmarks` | Benchmark summaries |
| `GOLDEN_DATASET_PATH` | `./datasets/golden_dataset.json` | Default goldens |

## Exit criteria

- DeepEval behind adapter; domain never imports DeepEval types in business logic
- LangSmith behind `Tracer` / `TracingService`; run id/url on `EvaluationReport`
- Tool validation framework ready for future agents
- Golden dataset loader + extended schema
- Evaluation / regression / benchmark CLIs
- Multi-format reports with required fields
- Mocked pytest coverage
- README + ADR-010

## Interview soundbite

> "Sprint 9 gave us a tool-independent Metric registry. Sprint 10 plugs in
> DeepEval as Strategies, LangSmith as a Tracer Strategy, and a custom tool
> validator for agent contracts. Regression and benchmark runners turn
> offline goldens into release gates — same idea as comparing JUnit XML
> across builds, but for RAG quality and future tool-using agents."

## Related ADR

[ADR-010 — AI Evaluation Automation](../architecture/ADR-010-AI-Evaluation-Automation.md)
