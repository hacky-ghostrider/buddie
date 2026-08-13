# Sprint 6 — Retrieval Layer

## Goal

Retrieve the most relevant stored chunks for a user question. No LLM
generation, prompts, RAG chains, FastAPI Q&A, or DeepEval in this sprint.

```text
Question  →  Query embedding  →  Similarity search  →  Top-K RetrievedDocument
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | What is retrieval? | Selecting the best supporting chunks for a question |
| 2 | Separate from generation | Fetch books first; write the essay later |
| 3 | Retrieval vs search vs generation | Orchestration vs ANN lookup vs answer writing |
| 4 | Why semantic retrieval works | Similar meaning → nearby vectors |
| 5 | Cosine similarity | Compare direction of meaning-arrows, ignore length |
| 6 | Top-K | Keep only the K nearest neighbors |
| 7 | Score threshold | Drop weak matches below a minimum similarity |
| 8 | Too many chunks | Noise crowds the LLM context window |
| 9 | Too few chunks | Missing facts → hallucination |
| 10 | Precision vs recall | Relevant among returned vs relevant among all |
| 11 | Dense vs sparse | Embeddings vs keyword/BM25 |
| 12 | Hybrid search | BM25 + vectors (future) |
| 13 | Metadata filtering | Constrain by file/page/type before/during ANN |
| 14 | No LLM in retriever | Swappable stores; measurable without generation |
| 15 | Interview staples | Cosine vs L2; Top-K tradeoffs; why abstract Retriever? |

### Analogies

- Retrieval ≈ a librarian fetching the most relevant books for your question.
- Search ≈ looking up those books by “meaning GPS” on the map (vector store).
- Generation ≈ writing a book report from the fetched books (later sprint).
- Top-K ≈ “give me the 5 closest pins,” not the whole city.
- Threshold ≈ “ignore pins more than X blocks away.”

## What we tried to achieve

- `Retriever` ABC (`retrieve`)
- `VectorRetriever` composing `EmbeddingModel` + `VectorStore`
- `RetrievedDocument` with `id`, `text`, `metadata`, `score`
- `VectorStore.similarity_search(query_embedding, top_k, score_threshold)`
- Settings: `TOP_K`, `DEFAULT_SCORE_THRESHOLD`
- Domain exceptions + centralized logging (no `print`)
- Pytest with deterministic fake embedder + disk-backed Chroma double
- Explicitly **no** LLM / prompt / RAG chain / eval APIs

## Architecture

```text
Question (str)
      │
      ▼
Retriever (ABC)                    app/retrieval/
      │
      ▼
VectorRetriever
      │  embed_query()
      ▼
EmbeddingModel (Sprint 4)
      │  query_embedding
      ▼
VectorStore.similarity_search()    app/vectorstore/
      │
      ▼
ChromaVectorStore (cosine space)
      │
      ▼
List[RetrievedDocument]  { id, text, metadata, score }
```

### Why document vs query embedding stay conceptually separate

Ingestion encodes corpus chunks once (`embed_documents`). Retrieval encodes a
live question per request (`embed_query`). Same model today; different
lifecycle, batching, and room for asymmetric models later.

### Why score matters

Score ranks candidates, enables thresholds, and feeds evaluation metrics
(hit-rate @K, MRR, nDCG) without re-embedding. Downstream eval can assert
“gold chunk was in top-3 with score ≥ 0.7.”

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/retrieval/base.py` | `Retriever` ABC | Vendor-agnostic retrieval contract |
| `app/retrieval/vector_retriever.py` | Dense retriever Strategy | Orchestrate embed + search |
| `app/retrieval/models.py` | `RetrievedDocument` | Explicit scored hit shape |
| `app/retrieval/exceptions.py` | Domain errors | Clear failure modes |
| `app/vectorstore/base.py` | `similarity_search` | Search belongs on the store contract |
| `app/vectorstore/chroma_store.py` | Cosine query + score convert | Chroma stays behind the ABC |
| `app/config/settings.py` | `TOP_K`, `DEFAULT_SCORE_THRESHOLD` | No hardcoded retrieval knobs |
| `tests/test_retrieval.py` | Suite | Top-k, threshold, determinism, metadata |
| `.env.example` / `README.md` | Docs | How to configure and reason about retrieval |
| `docs/architecture/ADR-006-*.md` | ADR | Why Retriever + store search split |

## Configuration

| Variable | Default | Validation |
|----------|---------|------------|
| `TOP_K` | `5` | Must be `> 0` |
| `DEFAULT_SCORE_THRESHOLD` | `0.0` | Must be in `[0, 1]` (cosine-normalized) |

## Deterministic retrieval for evaluation

Eval regressions need the **same question → same ranked ids/scores** under a
fixed embedder + store. Non-determinism makes “did retrieval get worse?”
unanswerable.

## Explicitly out of scope

LLM generation, prompt templates, RAG chains, Q&A endpoints, DeepEval,
hybrid BM25+dense (documented as future).

## Exit criteria

- Retrieve top-k scored documents  
- Empty query / empty store / missing collection handled  
- Metadata preserved; scores ordered descending  
- Threshold filtering works  
- Settings validate `TOP_K` and threshold  
- Deterministic retrieval tested  
- No generation APIs added  

## Interview soundbite

> "I keep retrieval separate from generation. `VectorRetriever` embeds the
> query through `EmbeddingModel`, searches via `VectorStore.similarity_search`,
> and returns scored `RetrievedDocument`s. The store never talks to an LLM —
> that keeps evaluation honest and backends swappable."
