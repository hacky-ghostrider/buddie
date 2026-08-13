# ADR-003: Chunking as an ingestion Strategy

- **Status:** Accepted  
- **Date:** 2026-08-02  
- **Sprint:** 3  

## Context

Loaded pages are often too large (or poorly bounded) for embedding and retrieval.
Chunking must be configurable, testable, and independent of PDF loaders and
vector stores. Placing chunking under a generic `app/chunking` top-level package
would hide that it belongs to the **ingestion lifecycle** (before embeddings).

## Decision

1. Place chunking under `app/ingestion/chunking/` beside `app/ingestion/loaders/`.
2. Define a `Chunker` ABC with `chunk(documents) -> list[Document]`.
3. Implement `RecursiveChunker` using `RecursiveCharacterTextSplitter`.
4. Drive `chunk_size`, `chunk_overlap`, and `separators` from validated settings.
5. Enrich chunks with deterministic metadata (`chunk_id`, `chunk_index`,
   `total_chunks`, `chunk_size`) while preserving loader metadata.
6. Keep `DocumentIngestionService.load()` load-only; optional composition via
   injected `Chunker` / `load_and_chunk()`.

## Why this package layout?

```text
Source → Loader → Chunker → Embeddings → Vector Store → Retriever → LLM → Eval
```

Grouping loaders + chunking under `ingestion/` mirrors the data lifecycle and
keeps retrieval/eval packages free of “how we sliced the text.”

## Consequences

- **Positive:** Clear SRP; easy to add token/semantic chunkers later.  
- **Positive:** Deterministic ids support regression + eval baselines.  
- **Negative:** One-time move of `app/loaders` → `app/ingestion/loaders` (compat shim retained).  
- **Deferred:** Async chunking; semantic/token Strategies; embedding stage (Sprint 4+).
