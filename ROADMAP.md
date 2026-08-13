# Roadmap

## Version 1.0 (current) — shipped

- Production RAG pipeline with Strategy abstractions
- LangGraph single agent (planner / router / tools)
- DeepEval + LangSmith adapters
- Tool contracts and validation
- Continuous evaluation quality gates
- Benchmark history + regression
- Docker, Makefile, GitHub Actions
- One-command offline demo

Architecture is **frozen**. No v2 features in this branch.

---

## Version 2 — planned (not implemented)

| Theme | Items |
|-------|--------|
| Agent intelligence | Memory, Reflection, multi-agent / supervisor patterns |
| Tooling ecosystem | MCP (Model Context Protocol) integrations |
| Evaluation vendors | RAGAS adapters, Phoenix observability |
| Observability | OpenTelemetry traces/metrics/logs |
| Scale | Distributed evaluation workers, larger golden suites |
| Platforms | CrewAI / AutoGen adapters behind existing tool contracts |

### Principles for v2

1. Keep `Metric`, `Tracer`, `ToolContract`, `EvaluationContext`, and quality gates stable.
2. Add adapters — do not fork evaluation for each vendor.
3. Multi-agent only after single-agent evaluation SLOs are proven in production.

---

## Explicitly out of scope until v2

MCP · Memory · Reflection · Multi-Agent · CrewAI · AutoGen · Phoenix · RAGAS · new evaluation metric families · architectural rewrites.
