# Sprint 2 — Document Ingestion Layer

## Goal

Load and validate documents through a **format-agnostic** ingestion layer. PDF is the first implementation — not the architecture.

## What we tried to achieve

- `DocumentLoader` ABC (Strategy)
- `PDFDocumentLoader` using LangChain `PyPDFLoader` (raw pages only)
- Domain exceptions for failure modes
- Normalized metadata for evaluation provenance
- Settings-driven allowed extensions
- Comprehensive pytest coverage (real + mocked)

## Why

RAG evaluation needs reliable, traceable source text. Coupling everything to `PyPDFLoader` would block Word/Markdown/Confluence/S3 later and make testing painful. Abstraction now = cheaper extension later.

See [ADR-001](../architecture/ADR-001-document-loader-abstraction.md).

## What we did (summary)

| Area | Outcome |
|------|---------|
| Contract | `DocumentLoader.load() -> list[Document]` |
| PDF | Validate path/ext/permissions; load pages; enrich metadata |
| Errors | Not found, unsupported type, empty, corrupt, permission |
| Config | `SUPPORTED_DOCUMENT_TYPES` |
| Service | Initial orchestration entry point |
| Tests | Happy path + all major failure modes |
| Samples | `data/README.md` (no large PDFs in git) |

## File changes and why

| File | Change | Why |
|------|--------|-----|
| `app/loaders/base.py` | ABC | Depend on contract, not PDF |
| `app/loaders/pdf_loader.py` | PDF strategy | First concrete loader; injectable `PyPDFLoader` for tests |
| `app/loaders/exceptions.py` | Domain errors | Stable, mappable failure types |
| `app/loaders/constants.py` | Metadata string constants | Avoid magic strings (evolved in 2.1 to enum) |
| `app/services/document_ingestion_service.py` | Use-case entry | Callers don’t construct PDF loader directly |
| `app/config/settings.py` | `supported_document_types` | Config-driven allow-list |
| `.env.example` | Document new env var | Discoverability |
| `tests/conftest.py` | PDF fixtures | Reusable temp PDFs / samples |
| `tests/test_pdf_loader.py` | Ingestion tests | Prove validation + metadata |
| `data/README.md` | Sample PDF instructions | Local experimentation without committing binaries |
| `README.md` | Ingestion section | How it works + how to extend |
| `pyproject.toml` | `langchain-*`, `pypdf` | Runtime deps for PDF loading |

## Explicitly out of scope

Chunking, embeddings, vector DBs, retrieval, LLM calls, LangChain chains.

## Known follow-ups (became Sprint 2.1)

- Factory logic lived inside the service (`create_loader`) — separate it.
- Prefer `MetadataKeys` enum over string constants.
- Prefer extension **registry** over `if extension == ".pdf"`.

## Exit criteria

- Load a valid PDF → `list[Document]` with `source` / `page` / `file_name`  
- Failure modes raise domain exceptions  
- `pytest` covers success + failures  
