# ADR-006: Retrieval layer with query embedding + vector search

- **Status:** Accepted  
- **Date:** 2026-08-03  
- **Sprint:** 6  

## Context

Sprint 5 persists embeddings. Callers need a way to turn a natural-language
question into the best supporting chunks **without** coupling to LLMs,
prompts, or evaluation frameworks.

Gluing Chroma `query` calls into application services would leak vendor APIs
and mix orchestration (retrieve) with storage (ANN search).

## Decision

1. Place retrieval orchestration in top-level `app/retrieval/`.
2. Define `Retriever` with `retrieve(query) → list[RetrievedDocument]`.
3. Implement `VectorRetriever` that depends on:
   - `EmbeddingModel.embed_query` for the query vector
   - `VectorStore.similarity_search` for ANN lookup
4. Extend `VectorStore` with `similarity_search(query_embedding, top_k,
   score_threshold, metadata_filter?)` returning scored hits.
5. Convert Chroma cosine **distance** to similarity score `1 - distance`.
6. Configure defaults via `TOP_K` and `DEFAULT_SCORE_THRESHOLD`.
7. Keep generation, prompts, and evaluation out of this layer.

## Why this split?

| Layer | Responsibility |
|-------|----------------|
| `EmbeddingModel` | Text → vector (transform) |
| `VectorStore` | Persist + ANN search (storage) |
| `Retriever` | Question → top-k chunks (use case) |

Analogy: GPS encoder vs map board vs librarian.

## Consequences

- **Positive:** Retrievers stay LLM-free and testable with fake embedders.  
- **Positive:** Chroma remains an interchangeable Strategy.  
- **Positive:** Scores enable thresholds and later retrieval metrics.  
- **Negative:** Dense-only for now; hybrid BM25+vector deferred.  
- **Deferred:** Rerankers, multi-query retrieval, generation chains.
