# Sprint 4 — Embedding Engine

## Goal

Transform chunk/page `Document` objects into **vectors** via a reusable embedding
layer. No vector databases, retrieval, LLMs, or evaluation in this sprint.

```text
List[Document]  →  List[EmbeddedDocument]
```

## First principles (callouts)

| # | Topic | Callout |
|---|-------|---------|
| 1 | What is an embedding? | A list of numbers (a vector) that represents *meaning* of text |
| 2 | Why embeddings exist | Computers compare numbers easily; meaning needs a numeric form |
| 3 | LLM vs embedding model | LLM *generates* text; embedding model *maps* text → vector |
| 4 | Why vectors? | Geometry lets us measure “near” = “similar meaning” |
| 5 | Nearby vectors | Synonyms / related phrases land close in vector space |
| 6 | Cosine similarity | Angle between vectors (direction), ignore magnitude |
| 7 | Euclidean distance | Straight-line distance between points |
| 8 | Prefer cosine | Text length inflates magnitude; angle is stabler for semantics |
| 9 | Dimensions 384/768/1024 | Length of the vector — model capacity / cost trade-off |
| 10 | Good embedding model | Strong semantic retrieval, domain fit, speed, license, size |
| 11 | Small vs large | Small = fast/cheap/weaker; large = richer/slower/heavier |
| 12 | Independent of vector DB | Embedding = transform; store/index = persistence — separate SRP |
| 13 | Interview staples | Cosine vs L2; why normalize; why batch; why not hardcode model |

### Analogies

- Embedding ≈ GPS coordinates for a sentence’s meaning.
- Cosine ≈ comparing *direction* of two arrows, not how long they are.
- Batching ≈ baking cookies on a tray instead of one-by-one — GPU/CPU work amortizes.

## What we tried to achieve

- `EmbeddingModel` ABC (`embed_documents`, `embed_query`)
- `SentenceTransformerEmbedding` with configurable `BAAI/bge-small-en-v1.5`
- `EmbeddedDocument` (text + embedding + metadata) — no parallel list footguns
- Configurable batching + optional L2 normalization
- Domain exceptions + centralized logging
- Pytest with mocked model (CI-friendly; no weight download required)
- Folder evolution toward lifecycle packages (`embeddings/`, placeholders for `vectorstore/`, `retrieval/`, `generation/`, `evaluation/`)

## Architecture

```text
Ingestion (loaders → chunkers)
        │  List[Document]
        ▼
EmbeddingModel (ABC)                 app/embeddings/
        │
        ▼
SentenceTransformerEmbedding
        │
        ▼
List[EmbeddedDocument]  { text, embedding, metadata }
        │
        ▼
Vector Store / Retrieval / Generation / Evaluation   # later sprints
```

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/embeddings/base.py` | `EmbeddingModel` ABC | Strategy contract independent of vendors |
| `app/embeddings/sentence_transformer_embedding.py` | ST implementation (lazy import) | Local baseline; injectable factory for tests |
| `app/embeddings/models.py` | `EmbeddedDocument` | Keep text↔vector↔metadata aligned |
| `app/embeddings/exceptions.py` | Domain errors | Clear failure modes for APIs/jobs |
| `app/config/settings.py` | `EMBEDDING_MODEL`, `EMBED_BATCH_SIZE`, `NORMALIZE_EMBEDDINGS` | No hardcoded model/batch |
| `app/vectorstore\|retrieval\|generation\|evaluation/` | Placeholder packages | Mirror mature AI platform layout |
| `app/ingestion/chunking/recursive_splitter.py` | In-house recursive splitter | Avoid `langchain_text_splitters` → torch import chain |
| `app/ingestion/loaders/pdf_loader.py` | `pypdf` page loader (PyPDFLoader-compatible docs) | Same Document shape without community import toxicity |
| `tests/conftest.py` | Optional torch/ST stubs | Keep CI green when native torch DLLs fail |
| `tests/test_embeddings.py` | Suite | Dimension, metadata, batching, determinism, failures |
| `.env.example` / `README.md` | Docs | How to configure and reason about embeddings |
| `docs/architecture/ADR-004-*.md` | ADR | Why ST + why separate from vector DB |

## Configuration

| Variable | Default | Validation |
|----------|---------|------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Non-empty |
| `EMBED_BATCH_SIZE` | `32` | `> 0` |
| `NORMALIZE_EMBEDDINGS` | `true` | bool — preferred for cosine |

## Why `EmbeddedDocument` (not raw lists)?

Returning `list[list[float]]` alongside `list[Document]` invites index bugs
(“metadata for chunk 3 applied to vector 4”). One object = one atomic unit for
stores, retrieval, and evaluation.

## Why batching?

Each encode call has fixed overhead (kernel launch / graph setup). Batching
amortizes that cost across many texts — critical for large corpora.

## Why deterministic embeddings matter

Eval baselines and regression tests need the **same text → same vector** under
fixed model + settings. Non-determinism makes “did retrieval get worse?” unanswerable.

## Explicitly out of scope

Chroma / FAISS / Pinecone, similarity search APIs, RAG chains, LLMs, evaluation metrics.

## Production callout — dependency weight

`sentence-transformers` pulls **torch**. On some Windows setups native DLL load can fail.

Mitigations in this repo:

- Embedding code **lazy-imports** SentenceTransformer.
- Chunking uses an **in-house** `RecursiveCharacterSplitter` (does not import `langchain_text_splitters` package root, which eagerly pulls ST/torch).
- Pytest stubs `torch` / `sentence_transformers` unless `RAG_USE_REAL_EMBEDDINGS=1`.

Real embedding still requires a working torch install when you call the live model.

## Exit criteria

- Embed docs → `EmbeddedDocument` with preserved metadata  
- Embed query → vector of correct dimension  
- Settings validate model/batch  
- Tests green without downloading weights (mocked)  
- Sprint + ADR docs updated  

## Interview soundbite

> "An embedding model maps text to a vector so we can measure semantic closeness. I keep that transform behind `EmbeddingModel`, return `EmbeddedDocument` so text and vectors never desync, and leave persistence to a separate vector-store stage."
