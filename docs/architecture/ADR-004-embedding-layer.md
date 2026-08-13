# ADR-004: Embedding layer independent of vector stores

- **Status:** Accepted  
- **Date:** 2026-08-02  
- **Sprint:** 4  

## Context

Chunks must become dense vectors before indexing. Teams often glue
`SentenceTransformer.encode` directly into Chroma/FAISS calls, which couples
**transform** (embedding) to **persistence** (vector DB) and makes swaps hard
(OpenAI embeddings tomorrow, different store next month).

## Decision

1. Place embeddings in top-level `app/embeddings/` (lifecycle stage, not under ingestion).
2. Define `EmbeddingModel` with `embed_documents` / `embed_query`.
3. Implement `SentenceTransformerEmbedding` using settings-driven
   `EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`).
4. Return `EmbeddedDocument` (text + embedding + metadata).
5. Support configurable batching and optional L2 normalization.
6. Reserve empty packages: `vectorstore/`, `retrieval/`, `generation/`, `evaluation/`.

## Why Sentence-Transformers + bge-small?

- Strong open retrieval baseline for English  
- Runs locally (good for demos / offline eval)  
- Small footprint vs large embedding models  
- Easy to swap via `EMBEDDING_MODEL` without code edits  

## Consequences

- **Positive:** Clear SRP; testable with injected fakes; interview-friendly stages.  
- **Positive:** Normalization aligns with cosine similarity usage later.  
- **Negative:** `sentence-transformers` pulls heavy deps (torch).  
- **Deferred:** Vector store indexing, ANN search, remote embedding APIs.
