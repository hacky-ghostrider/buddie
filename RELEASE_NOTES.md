# Release Notes — v1.0.0

**Release date:** 2026-08-07  
**Codename:** Continuous AI Evaluation Platform (public GitHub release)

## Highlights

Version 1.0 freezes the architecture delivered across Sprints 1–12 and adds the
release engineering needed for a professional open-source launch:

- One-command offline demo (`make demo`) for interviews and recruiters
- Docker image + Compose with persistent Chroma and health checks
- GitHub Actions for lint, tests, evaluation, benchmarks, quality gates, Docker
- Public documentation: README rewrite, Mermaid diagrams, interview pack, trade-offs
- MIT license and community files (`CONTRIBUTING`, `SECURITY`, `CODE_OF_CONDUCT`)

## What you can demonstrate

```text
Clone → make setup → make demo
        → LangGraph Agent
        → LangSmith (NoOp or live)
        → DeepEval metrics
        → Tool Validation
        → Quality Gates (PASS/WARNING/FAIL)
        → Reports under data/demo/
```

## Not included (by design)

MCP · Memory · Reflection · Multi-Agent · CrewAI · AutoGen · Phoenix · RAGAS ·
new evaluation metrics · architectural rewrites.

See [ROADMAP.md](ROADMAP.md) for Version 2.

## Upgrade notes

- Package version is `1.0.0` (`app.__version__`, `pyproject.toml`)
- Copy fresh `.env.example` if you need Sprint 12 quality-gate variables
- Prefer `make demo` over ad-hoc script invocation for the canonical path
