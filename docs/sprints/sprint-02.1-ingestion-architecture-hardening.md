# Sprint 2.1 — Ingestion Architecture Hardening

## Goal

Refine Sprint 2 architecture for **SRP, extensibility, and interview clarity** — without adding new document formats or RAG pipeline stages.

## What we tried to achieve

1. **Remove factory logic from the service** — `DocumentLoaderFactory` owns construction; `DocumentIngestionService` only orchestrates `load()`.
2. **`MetadataKeys` enum** — `metadata[MetadataKeys.SOURCE]` instead of typo-prone strings.
3. **Loader registry** — `{" .pdf": builder, ...}` instead of `if extension == ".pdf"`.
4. **Document ADRs + sprint docs** — capture *why*, not only *what*.
5. **Note async for later** — enterprise IO-bound ingestion; not implemented yet.

## Why

Sprint 2 was correct functionally but mixed responsibilities. Separating factory/registry makes “add Word/Markdown” a registration problem, not a service rewrite. ADRs freeze the interview-ready rationale.

See [ADR-002](../architecture/ADR-002-loader-factory-and-registry.md).

## Architecture (after 2.1)

```text
API (future)
    │
    ▼
DocumentIngestionService
    │
    ▼
DocumentLoaderFactory
    │   registry = { ".pdf": PDF..., ".md": ..., ".docx": ... }
    ▼
Concrete DocumentLoader (Strategy)
    │
    ▼
List[LangChain Document]  +  MetadataKeys
    │
    ▼
Validation / domain exceptions
    │
    ▼
Pipeline (chunk → embed → retrieve → evaluate)  # later
```

## What we did (summary)

| Change | Detail |
|--------|--------|
| Factory | New `app/loaders/factory.py` with `DEFAULT_LOADER_REGISTRY` + `register()` |
| Service | Injects `DocumentLoaderFactory`; no `create_loader` / `if` branches |
| Metadata | `MetadataKeys(StrEnum)`; aliases kept in `constants.py` |
| Docs | `docs/sprints/*`, `docs/architecture/ADR-001`, `ADR-002` |
| Tests | Factory, registry extension, service DI with fake registry |

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/loaders/factory.py` | **Added** factory + registry | Construction ≠ orchestration (SRP) |
| `app/loaders/metadata_keys.py` | **Added** `MetadataKeys` | Type-safe metadata keys |
| `app/loaders/constants.py` | Re-export aliases | Backward compatible with Sprint 2 imports |
| `app/loaders/pdf_loader.py` | Use `MetadataKeys.*` | Consistent key access |
| `app/loaders/__init__.py` | Export factory + enum | Clean public API |
| `app/services/document_ingestion_service.py` | Depend on factory via DI | Thin service layer |
| `tests/test_pdf_loader.py` | Factory/registry/service DI tests | Prove extensibility without new formats |
| `docs/README.md` | Docs index | Navigate sprints + ADRs |
| `docs/sprints/sprint-01-*.md` | Sprint 1 retrospective | Teaching / interview trail |
| `docs/sprints/sprint-02-*.md` | Sprint 2 retrospective | Teaching / interview trail |
| `docs/sprints/sprint-02.1-*.md` | This document | Capture hardening rationale |
| `docs/architecture/ADR-001-*.md` | Abstraction ADR | “Why not PyPDFLoader everywhere?” |
| `docs/architecture/ADR-002-*.md` | Factory/registry ADR | “Why separate factory?” |
| `README.md` | Updated diagrams + links | Keep root README aligned |

## Explicitly deferred

| Item | When |
|------|------|
| Word / HTML / Markdown / Confluence / S3 / Notion loaders | Later sprints |
| `async load(...)` | When IO-bound scale requires it |
| Chunking / embeddings / retrieval / LLM | Sprint 3+ |

## How to add a new loader (after 2.1)

1. Implement `DocumentLoader` in `app/loaders/<type>_loader.py`.
2. Add a builder to `DEFAULT_LOADER_REGISTRY` (or call `factory.register(".ext", builder)`).
3. Allow the extension in `SUPPORTED_DOCUMENT_TYPES`.
4. Use `MetadataKeys` when writing metadata.
5. Add tests (happy path + failures).

## Exit criteria

- Service has no extension `if/else`  
- Factory registry resolves `.pdf`  
- Metadata written via `MetadataKeys`  
- ADRs + sprint docs exist under `docs/`  
- `pytest` green  

## Interview soundbite

> "`PyPDFLoader` is an implementation detail. The pipeline depends on `DocumentLoader`. A factory + registry selects the concrete strategy by extension. The service only runs the use case. That keeps ingestion extensible and testable."
