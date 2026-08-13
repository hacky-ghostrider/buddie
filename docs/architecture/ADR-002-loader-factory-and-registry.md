# ADR-002: Loader factory, registry, and MetadataKeys

- **Status:** Accepted  
- **Date:** 2026-08-02  
- **Sprint:** 2.1  

## Context

Sprint 2 placed loader selection (`if extension == ".pdf"`) inside `DocumentIngestionService.create_loader()`. That mixed **orchestration** with **construction**, violating Single Responsibility and making new formats require editing the service.

Metadata used string constants (`"source"`, `"page"`). Typos in dict keys are silent bugs.

## Decision

1. Introduce `DocumentLoaderFactory` with an **extension → builder registry**.
2. `DocumentIngestionService` **depends on the factory** (constructor injection); it no longer owns `if/else` creation logic.
3. Introduce `MetadataKeys(StrEnum)` and use `MetadataKeys.SOURCE` (etc.) instead of raw strings.
4. Keep sync `load()` for now; document **async** as a future enhancement for IO-bound enterprise ingestion.

## Registry shape

```python
DEFAULT_LOADER_REGISTRY = {
    ".pdf": _build_pdf_loader,
    # ".docx": _build_docx_loader,  # future
    # ".md": _build_markdown_loader,  # future
}
```

Adding a format = implement loader + `register(...)` (or extend the default map) + allow extension in settings. No service rewrite.

## Target call flow

```text
API (future)
  → DocumentIngestionService
    → DocumentLoaderFactory (registry)
      → Concrete DocumentLoader
        → LangChain Document (+ MetadataKeys)
          → Validation / domain exceptions
            → Pipeline (later)
```

## Async (deferred)

Enterprise ingestion is often IO-bound (network shares, S3, many files). A future `async def load(...)` (and async loaders) will unlock concurrency without changing the Strategy contract’s meaning. Sync is correct until that pressure exists.

## Consequences

- **Positive:** Service stays thin; factory is unit-testable with a fake registry.  
- **Positive:** Enum keys reduce metadata typos.  
- **Negative:** One more class to learn (`DocumentLoaderFactory`).  
- **Neutral:** Backward-compatible aliases remain in `constants.py` for older imports.
