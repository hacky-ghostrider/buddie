# Sprint 3 — Chunking Engine

## Goal

Convert loaded LangChain `Document` pages into high-quality **chunks** —
still `list[Document]` — with enriched metadata. No embeddings, vector DBs,
retrieval, LLMs, or evaluation in this sprint.

## What we tried to achieve

- A reusable `Chunker` Strategy independent of PDF / embeddings / vector stores
- `RecursiveChunker` via LangChain `RecursiveCharacterTextSplitter`
- Config-driven `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SEPARATORS` with validation
- Preserve Sprint 2 metadata; add `chunk_id`, `chunk_index`, `total_chunks`, `chunk_size`
- Reorganize packages under `app/ingestion/` (loaders + chunking) to mirror the data lifecycle
- Comprehensive pytest coverage including **deterministic** output

## Why chunking matters (callouts)

| Topic | Callout |
|-------|---------|
| Importance | Retrieval quality is often decided here — more than the LLM choice |
| Bad chunking | Lost context, cut sentences, noisy neighbors, weak citations |
| Separation from loaders | Loaders know *sources*; chunkers know *text windows* — different SRP |
| Configurable | Policy docs ≠ chat logs ≠ code; one size never fits all |
| Small chunks | Precise retrieval, weaker surrounding context |
| Large chunks | Richer context, more noise / token cost |
| Overlap | Keeps boundary sentences intact across windows |
| Strategy pattern | Swap recursive / token / semantic chunkers without rewriting the pipeline |

### Real-world examples

- **`chunk_size=1000`**: aim for ~1000 characters per chunk (like a short section).
- **`chunk_overlap=200`**: last ~200 chars of chunk N start chunk N+1 — so a sentence split at the boundary still appears whole in at least one chunk.

## Architecture

```text
Document Source
      │
      ▼
Document Loader          (app/ingestion/loaders)
      │  List[Document]  # pages
      ▼
Chunker                  (app/ingestion/chunking)
      │  List[Document]  # chunks + MetadataKeys
      ▼
Embedding / Vector Store / Retriever / LLM / Eval   # later sprints
```

### Package layout (Sprint 3 decision)

```text
app/ingestion/
├── metadata_keys.py          # shared loader + chunk keys
├── loaders/                  # moved from app/loaders (shim kept)
└── chunking/
    ├── base.py               # Chunker ABC
    ├── recursive_chunker.py  # RecursiveChunker
    ├── metadata.py           # chunk metadata helpers
    └── exceptions.py
```

`app/loaders/` remains a **compat re-export** so older imports still work.

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/ingestion/` | **Added** package | Reflect ingestion lifecycle (load → chunk → …) |
| `app/ingestion/loaders/*` | Moved from `app/loaders` | Loaders are an ingestion stage |
| `app/ingestion/chunking/base.py` | `Chunker` ABC | Strategy contract: `chunk(docs) -> docs` |
| `app/ingestion/chunking/recursive_chunker.py` | Recursive implementation | First production Strategy |
| `app/ingestion/chunking/metadata.py` | Enrichment helpers | SRP — keep chunker focused on splitting |
| `app/ingestion/chunking/exceptions.py` | Domain errors | Stable failure types |
| `app/ingestion/metadata_keys.py` | Extended enum | Shared provenance vocabulary |
| `app/config/settings.py` | Chunk fields + validators | No hardcoded sizes; fail fast on bad config |
| `app/services/document_ingestion_service.py` | Optional `chunker` + `load_and_chunk` | Composition without forcing chunking on `load` |
| `app/loaders/__init__.py` | Compat shim | Avoid breaking Sprint 2 import paths |
| `tests/test_chunker.py` | Chunking suite | Overlap, metadata, determinism, invalid config |
| `.env.example` | Chunk env vars | Documented defaults |
| `docs/architecture/ADR-003-*.md` | Chunking ADR | Why separate + recursive strategy |
| `README.md` | Chunking section | How it works + example 1 page → N chunks |

## Metadata usefulness

| Field | Debugging | Retrieval | Evaluation | Citations |
|-------|-----------|-----------|------------|-----------|
| `chunk_id` | Trace a bad answer to a unit | Dedup / cache keys | Stable fixture ids | Point at exact span |
| `chunk_index` | Order within run | Reconstruct neighbors | Coverage metrics | “chunk 3 of page 2” |
| `total_chunks` | Spot explosion/collapse | Batching hints | Completeness checks | Context for reviewers |
| `chunk_size` | Find oversized outliers | Budget tokens | Correlate size vs score | — |
| Preserved `source`/`page`/`file_name` | Root cause | Filters | Golden-set joins | Human-readable cites |

## Determinism callout

Same input documents + same settings must yield the **same** chunk texts and `chunk_id`s.
That makes regression tests and evaluation baselines trustworthy. Random UUIDs for `chunk_id` would break that — we use stable `source::p{page}::c{index}` ids instead.

## Explicitly out of scope

Embeddings, vector databases, retrieval, LLMs, RAG chains, evaluation metrics.

## Exit criteria

- `RecursiveChunker.chunk(docs)` returns enriched chunks  
- Settings reject invalid size/overlap/separators  
- Tests cover empty input, metadata, overlap, determinism  
- Sprint + ADR docs updated  
- `pytest` green  

## Interview soundbite

> "Chunking is where RAG quality is often won or lost. I keep it as a Strategy behind `Chunker`, configure size/overlap/separators via settings, preserve source metadata, and add stable chunk ids so evaluation and citations stay reproducible."
