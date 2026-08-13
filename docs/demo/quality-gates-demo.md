# Interview Demo — Quality Gates & Continuous Evaluation

**Scenario id:** continuous quality decision after agent + evaluation  
**Sprint:** 12

## Walkthrough

```text
Question
   ↓
Agent (LangGraph Planner → Router → Tools)
   ↓
Evaluation (EvaluationContext + EvaluationReport)
   ↓
DeepEval metrics (faithfulness, hallucination, relevancy, …)
   ↓
LangSmith trace URL
   ↓
Tool Validation (ToolContract vs ToolExecution)
   ↓
Quality Gate Engine (configurable thresholds)
   ↓
PASS / WARNING / FAIL  (+ recommendations)
   ↓
Benchmark History (append + trend)
```

## What to show

1. **Agent run** produces `ToolExecution`s and `EvaluationContext` (Sprint 11).
2. **Evaluation automation** scores with DeepEval adapters and validates tools
   (Sprint 10) — still the same contracts.
3. **QualityGateEngine** applies rules from Settings:
   - `MIN_FAITHFULNESS`, `MAX_HALLUCINATION`, `MIN_RELEVANCY`, …
   - `MAX_TOOL_LATENCY`, `MAX_TOTAL_LATENCY`, `MAX_COST`
   - `PASS_THRESHOLD` / `WARNING_THRESHOLD`
4. **QualityDecision** records status, failed rules, warnings, correlation id,
   and actionable recommendations (e.g. low faithfulness → increase Top-K).
5. **BenchmarkHistory** stores the suite snapshot; comparison shows improving /
   degrading / stable trends.
6. Artifacts: `quality_report.json`, `quality_report.csv`, `quality_report.html`.

## Minimal code path (mocked tests)

```python
from app.evaluation.continuous import ContinuousEvaluationService
from app.evaluation.quality import QualityStatus

service = ContinuousEvaluationService(settings=settings)
result = service.evaluate(reports, suite_name="demo", write_reports=True)
assert result.decision.status in {
    QualityStatus.PASS,
    QualityStatus.WARNING,
    QualityStatus.FAIL,
}
```

## How this demonstrates enterprise AI Quality Engineering

| Enterprise need | Sprint 12 answer |
|-----------------|------------------|
| Release gates | PASS / WARNING / FAIL from policy, not vibes |
| Configurable SLOs | Thresholds from env / Settings |
| Prevent silent regressions | RegressionEngine + BenchmarkHistory |
| Auditability | Correlation id, LangSmith URL, structured reports |
| Actionability | Recommendations per failed rule |
| Continuity | Same EvaluationContext / ToolContract / ToolExecution |

This is the AI-evaluation analogue of **CI quality gates + historical
test dashboards**. Docker, GitHub Actions, and the one-command demo ship
in Sprint 13 / v1.0.
