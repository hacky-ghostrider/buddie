# Sprint 5 — Vector Store Layer (ChromaDB)

## Goal

Persist `List[EmbeddedDocument]` into a vector database behind a reusable
abstraction. No similarity search, retrieval, LLMs, or evaluation in this sprint.

```text
List[EmbeddedDocument]  →  VectorStore.add_documents()  →  document IDs / status
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | Vector database | Specialized store for high-dimensional float vectors + metadata |
| 2 | Why not SQL alone | B-tree / exact SQL filters are not optimized for “nearest vector” at scale |
| 3 | What it stores | id + embedding + optional text document + metadata |
| 4 | Embedding vs store vs retriever | Model *creates* vectors; store *keeps* them; retriever *queries* them |
| 5 | ANN | Approximate Nearest Neighbor — fast “close enough” search via indexes |
| 6 | Why indexing | Brute force is O(N); indexes trade a little recall for huge speed |
| 7 | No language understanding | Store compares numbers; meaning lived in the embedding model |
| 8 | Chroma architecture | Client → collection → records (ids/embeddings/documents/metadatas) + local persist |
| 9 | Chroma / FAISS / Pinecone / Weaviate / Qdrant | Local demo vs research index vs managed cloud vs hybrid search engines |
| 10 | Interview staples | Why abstract VectorStore? Why IDs? Why metadata separate? ANN vs kNN? |

### Analogies

- Vector DB ≈ a map of GPS pins (vectors) with sticky notes (metadata).
- SQL alone ≈ looking up phone numbers by name — great for exact keys, poor for “who is nearby in meaning-space”.
- ANN ≈ asking a librarian for the aisle of related books instead of reading every spine.
- Embedding model ≈ GPS encoder; vector store ≈ the pin board; retriever ≈ “find pins near this query”.

## What we tried to achieve

- `VectorStore` ABC (`create_collection`, `add_documents`, `delete_documents`,
  `delete_collection`, `collection_exists`, `count`)
- `ChromaVectorStore` with configurable collection + persist directory
- `VectorDocument` / `AddDocumentsResult` models
- Domain exceptions + centralized logging (no `print`)
- Settings: `VECTOR_DB`, `CHROMA_COLLECTION_NAME`, `CHROMA_PERSIST_DIRECTORY`
- Pytest coverage including persistence-across-reopen
- Explicitly **no** `query` / similarity search API in this sprint

## Architecture

```text
EmbeddingModel
      │  List[EmbeddedDocument]
      ▼
VectorStore (ABC)                 app/vectorstore/
      │
      ▼
ChromaVectorStore
      │
      ▼
Chroma collection (local persist)
      │
      ▼
AddDocumentsResult { ids, added_count, collection_name }
```

Embedding service knows nothing about Chroma. Retriever (Sprint 6) will know
nothing about how vectors are stored — only the `VectorStore` contract.

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/vectorstore/base.py` | `VectorStore` ABC | Vendor-agnostic persistence contract |
| `app/vectorstore/chroma_store.py` | Chroma Strategy + injectable client | Local persist; testable without native crashes |
| `app/vectorstore/models.py` | `VectorDocument`, `AddDocumentsResult` | Explicit stored shape + add outcome |
| `app/vectorstore/exceptions.py` | Domain errors | Clear failure modes |
| `app/config/settings.py` | Vector DB settings + validators | No hardcoded paths/names |
| `tests/test_vectorstore.py` | Suite + disk-backed fake client | Contract + persistence without flaky native deps |
| `.env.example` / `README.md` | Docs | How to configure and reason about stores |
| `docs/architecture/ADR-005-*.md` | ADR | Why Chroma + why abstract |

## Configuration

| Variable | Default | Validation |
|----------|---------|------------|
| `VECTOR_DB` | `chroma` | Must be `chroma` (only backend this sprint) |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Non-empty, ≥ 3 chars |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Non-empty path |

## Why IDs are important

Without stable ids you cannot delete, re-index, or cite a specific chunk.
Prefer ingestion `chunk_id` as the store id.

## Why metadata stays separate

Vectors answer “what is similar?”. Metadata answers “where did this come from?”
and enables filters (page, file, chunk index) without polluting geometry.

## Why persistence tests matter

Services restart. If reopen loses vectors, every downstream eval is a coin flip.
Persistence tests reopen a fresh client against the same directory and assert
`count()` still matches.

## Windows note (native Chroma)

Some Windows hosts hit an access violation inside Chroma’s Rust bindings on
`collection.add`. This repo:

- Implements the real `ChromaVectorStore` against Chroma’s public API
- Prefers SegmentAPI on Windows when `hnswlib` is available
- Runs contract/persistence tests via an injectable disk-backed client double
- Ships an optional `@pytest.mark.integration` smoke test that skips when native
  add is unsafe

Remediation for real native runs: Linux CI, Docker `HttpClient`, or install
MSVC build tools + `hnswlib` for SegmentAPI.

## Explicitly out of scope

Similarity-search orchestration / retrievers, RAG chains, LLMs, DeepEval.
(`similarity_search` on the store was deferred to Sprint 6.)

## Exit criteria

- Create / delete collection  
- Add / delete documents by id  
- Reject duplicate ids and bad dimensions  
- `count()` works  
- Persistence survives reopen (tested)  
- Settings validate vector DB config  
- No retrieval orchestration APIs in this sprint  

## Interview soundbite

> "I keep embedding creation separate from vector persistence. `VectorStore`
> stores ids, text, embeddings, and metadata. Chroma is one Strategy. Search
> belongs to the store's query API and the retriever's orchestration — not the
> write path."
