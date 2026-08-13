# Contributing

Thanks for improving the RAG Evaluation Platform.

## Ground rules

1. **Architecture is frozen at v1.0.** Do not add MCP, Memory, Reflection, Multi-Agent, CrewAI, AutoGen, Phoenix, RAGAS, or new metric families in this major version — propose them against [ROADMAP.md](ROADMAP.md) instead.
2. Keep vendor SDKs behind adapters (`Metric`, `Tracer`, `VectorStore`, `LLMProvider`, …).
3. Tests must mock vendors — no live OpenAI / LangSmith / DeepEval LLM calls in CI.
4. Prefer small, reviewable PRs with tests.

## Development setup

```bash
make setup
make test
make lint
make demo
```

Copy `.env.example` to `.env` only when you need live integrations.

## Pull requests

- Describe *why* the change exists
- Link related ADR / sprint doc when relevant
- Ensure `make ci` passes locally (lint + test + evaluate + quality-gate)
- Do not commit secrets, large PDFs, or `data/` runtime artifacts

## Code style

- Python 3.12+
- Ruff for lint (`make lint`)
- Type hints on public APIs
- Docstrings on modules and public classes/functions

## Reporting bugs

Open an issue with reproduction steps, expected vs actual behavior, and whether the run was offline (`make demo`) or live.
