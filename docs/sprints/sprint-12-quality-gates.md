# Sprint 12 — Continuous AI Evaluation / Quality Gates

**Status:** Complete  
**Depends on:** Sprint 11 (LangGraph Agent), Sprint 10 (Evaluation Automation)

## Goal

Transform the platform into a production-quality **Continuous AI Evaluation
Platform**. Every execution automatically determines **PASS / WARNING / FAIL**
using configurable quality gates, with regression detection and benchmark
history — without changing previous architecture.

## Delivered

| Item | Location |
|------|----------|
| Quality Gate Engine | `app/evaluation/quality/` |
| QualityDecision | `app/evaluation/quality/decision.py` |
| Configurable rules + thresholds | `app/evaluation/quality/models.py` + Settings |
| Benchmark History | `app/evaluation/benchmark/history.py` |
| Dashboard models (no web UI) | `app/evaluation/benchmark/dashboard.py` |
| Regression (incl. cost) | `app/evaluation/regression/runner.py` |
| Quality reports (JSON/CSV/HTML) | `app/evaluation/quality/report.py` |
| ContinuousEvaluationService | `app/evaluation/continuous.py` |
| EvaluationTimeline | `app/evaluation/timeline.py` |
| PlannerDecision | `app/agent/models.py` |
| ToolResult / ToolExecutionMetrics | `app/evaluation/tool_validation/` |
| Tests | `tests/test_quality_gates.py` |
| ADR | [ADR-012](../architecture/ADR-012-AI-Quality-Gates.md) |
| Interview demo | [quality-gates-demo.md](../demo/quality-gates-demo.md) |

## Architecture

```text
LangGraph Agent
       │
       ▼
Evaluation Framework
       │
┌──────┼──────────────┐
▼      ▼              ▼
DeepEval  Tool Validation  LangSmith
└──────┼──────────────┘
       ▼
Quality Gate Engine
       │
       ▼
Evaluation Decision
PASS / WARNING / FAIL
       │
       ▼
Benchmark History
```

## Configuration

| Env var | Purpose |
|---------|---------|
| `QUALITY_GATE_ENABLED` | Master switch |
| `MIN_FAITHFULNESS` | Min faithfulness |
| `MAX_HALLUCINATION` | Max hallucination |
| `MIN_RELEVANCY` | Min answer relevancy |
| `MIN_CONTEXT_PRECISION` | Min context precision |
| `MIN_CONTEXT_RECALL` | Min context recall |
| `MAX_TOOL_LATENCY` | Max tool latency (ms) |
| `MAX_TOTAL_LATENCY` | Max e2e latency (ms) |
| `MAX_COST` | Max estimated USD cost |
| `PASS_THRESHOLD` | Overall PASS cut-off |
| `WARNING_THRESHOLD` | WARNING band floor |

## Out of scope (delivered in Sprint 13 / v1.0)

Docker, GitHub Actions, Makefile demo packaging — see
[sprint-13-production-release.md](sprint-13-production-release.md).

Still out of scope for v1: MCP, Memory, Reflection, Supervisor, Multi-Agent,
CrewAI, AutoGen, Streaming, Human-in-the-loop.

## Exit criteria

- [x] Quality gates produce PASS / WARNING / FAIL  
- [x] Configurable thresholds from Settings  
- [x] Benchmark history + comparison  
- [x] Regression detects score / latency / tool / prompt / cost  
- [x] quality_report.json / .csv / .html  
- [x] Dashboard models (no web UI)  
- [x] PlannerDecision, ToolResult, ToolExecutionMetrics, EvaluationTimeline  
- [x] Actionable quality recommendations  
- [x] PyTest coverage with mocked vendors  
