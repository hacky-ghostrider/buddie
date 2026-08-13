# ADR-001: DocumentLoader abstraction

- **Status:** Accepted  
- **Date:** 2026-08-02  
- **Sprint:** 2  

## Context

We need to ingest documents into LangChain `Document` objects for a RAG evaluation platform. The first source is PDF (`PyPDFLoader`), but production systems rarely stay PDF-only. Hardcoding `PyPDFLoader` in services and APIs would couple the entire pipeline to one vendor API and one file format.

## Decision

1. Introduce a `DocumentLoader` abstract base class with `load() -> list[Document]`.
2. Implement `PDFDocumentLoader` as the first Strategy.
3. Keep a thin **service layer** (`DocumentIngestionService`) for the use-case entry point.
4. Raise **domain-specific exceptions** (`DocumentNotFoundError`, `CorruptedDocumentError`, etc.) instead of bare `Exception`.

## Why DocumentLoader abstraction?

`PyPDFLoader` is an **implementation detail**. The rest of the pipeline must not care whether text came from PDF, Word, Markdown, Confluence, SharePoint, or S3. Depend on the abstraction; swap concretes.

### Interview answer

> "Because `PyPDFLoader` is an implementation detail. I designed a `DocumentLoader` abstraction so the rest of the pipeline doesn't care whether documents come from PDFs, Word, Markdown, Confluence, SharePoint, or S3. The ingestion service depends on the abstraction rather than a concrete loader, making the system easier to extend and test."

## Why Strategy Pattern?

Each loader is an interchangeable algorithm for “turn a source into `list[Document]`”. Callers select a strategy (by extension / scheme) without branching on PDF internals.

## Why a Service Layer?

Separates HTTP/CLI/jobs from domain loading. The service owns the **use case** (“ingest this path”); loaders own **how**. This mirrors a Spring `@Service` that depends on interfaces, not `JdbcTemplate` internals.

## Why custom exceptions?

APIs and jobs need stable error categories (404-like not found, 415-like unsupported type, 422-like empty/corrupt). Generic exceptions hide intent and complicate mapping / testing.

## Consequences

- **Positive:** Extensible, testable, interview-clear architecture.  
- **Positive:** Failure modes are explicit.  
- **Negative:** More types/files than a one-off script.  
- **Follow-up:** Sprint 2.1 moves factory/`if` logic out of the service (see ADR-002).
