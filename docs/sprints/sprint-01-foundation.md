# Sprint 1 — Foundation

## Goal

Create a production-ready **project skeleton** for a RAG Evaluation Platform — without any RAG logic.

## What we tried to achieve

- A maintainable Python layout (easy to explain in interviews)
- Typed configuration and centralized logging
- A runnable FastAPI app with a health endpoint
- A minimal pytest smoke suite
- Tooling via `uv` + `pyproject.toml`

## Why

Java analogy: start with a Maven/Spring Boot skeleton (`pom.xml`, `application.yml`, logging config, `/actuator/health`) before writing business features. Same idea here — foundations first so later sprints plug into known packages.

## What we did (summary)

| Area | Outcome |
|------|---------|
| Packaging | `pyproject.toml`, `uv`, Python 3.12+ |
| Config | `Settings` via Pydantic Settings + `.env.example` |
| Logging | Central `setup_logging()` — no `print()` |
| API | `create_app()` factory + `GET /health` |
| Tests | Health + settings defaults |
| Docs | Root `README.md` |

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `pyproject.toml` | Created project + deps | Single dependency/source-of-truth (like `pom.xml`) |
| `.gitignore` | Ignore venv, `.env`, caches | Keep secrets and junk out of git |
| `.env.example` | Document env vars | Safe template; real `.env` stays local |
| `app/config/settings.py` | `BaseSettings` + `get_settings()` | Typed, validated, cached config |
| `app/config/logging.py` | Root logger setup | One place for format/level (Logback-style) |
| `app/main.py` | FastAPI + lifespan + `/health` | Runnable ASGI entry; testable factory |
| `app/api|core|models|rag|services|utils/` | Empty packages | Reserved namespaces for later sprints |
| `tests/test_health.py` | Smoke tests | Prove foundation works |
| `README.md` | How to run/test | Onboarding |

## Explicitly out of scope

PDF loading, chunking, embeddings, vector DBs, retrieval, LLMs.

## Exit criteria

- `uv sync` works  
- `uvicorn app.main:app` serves `/health`  
- `pytest` passes  
