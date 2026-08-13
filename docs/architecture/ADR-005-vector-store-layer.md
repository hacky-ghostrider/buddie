# ADR-005: Vector store abstraction with Chroma persistence

- **Status:** Accepted  
- **Date:** 2026-08-03  
- **Sprint:** 5  

## Context

Sprint 4 produces `EmbeddedDocument` values. They must be persisted before
retrieval. Gluing Chroma calls into the embedding layer would couple
**transform** (embedding) to **persistence** (vector DB) and later to
**query** (retriever).

## Decision

1. Place persistence in top-level `app/vectorstore/`.
2. Define `VectorStore` with collection lifecycle + `add_documents` /
   `delete_documents` / `count` (no similarity search in this sprint).
3. Implement `ChromaVectorStore` with settings-driven
   `CHROMA_COLLECTION_NAME` and `CHROMA_PERSIST_DIRECTORY`.
4. Resolve document ids from metadata (`chunk_id` preferred).
5. Keep metadata separate from embedding vectors.
6. Inject a Chroma client factory so tests can verify persistence without
   depending on fragile native bindings.

## Why Chroma?

- Local persistence suitable for demos and offline evaluation  
- Simple Python API (collections + ids/embeddings/documents/metadatas)  
- Easy swap later behind `VectorStore` (Qdrant / Pinecone / Weaviate)  
- Aligns with teaching incremental RAG platforms  

## Consequences

- **Positive:** Embedding and retrieval stay independent of Chroma details.  
- **Positive:** IDs enable delete/reindex/citation workflows.  
- **Negative:** Native Chroma on some Windows hosts is unstable (Rust `add`
  access violations); mitigate with SegmentAPI+hnswlib, Docker HTTP, or Linux.  
- **Deferred:** ANN query APIs, hybrid search, remote managed stores.
