"""Tests for RecursiveChunker and chunk metadata enrichment."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.config.settings import Settings
from app.ingestion.chunking.exceptions import (
    EmptyDocumentListError,
    NoChunkableContentError,
)
from app.ingestion.chunking.recursive_chunker import RecursiveChunker
from app.ingestion.metadata_keys import MetadataKeys


def _page(
    text: str,
    *,
    source: str = "/docs/sample.pdf",
    page: int = 0,
    file_name: str = "sample.pdf",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            MetadataKeys.SOURCE: source,
            MetadataKeys.PAGE: page,
            MetadataKeys.FILE_NAME: file_name,
        },
    )


@pytest.fixture
def chunk_settings() -> Settings:
    """Small chunk window so tests produce multiple chunks from short text."""
    return Settings(
        app_env="test",
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", " ", ""],
    )


def test_normal_documents_produce_chunks(chunk_settings: Settings) -> None:
    """Normal page text should yield one or more chunks with content."""
    text = " ".join(f"word{i}" for i in range(40))
    chunker = RecursiveChunker(settings=chunk_settings)

    chunks = chunker.chunk([_page(text)])

    assert len(chunks) >= 1
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_empty_document_list_raises(chunk_settings: Settings) -> None:
    """Empty input list must raise EmptyDocumentListError."""
    chunker = RecursiveChunker(settings=chunk_settings)

    with pytest.raises(EmptyDocumentListError):
        chunker.chunk([])


def test_empty_page_content_raises(chunk_settings: Settings) -> None:
    """Whitespace-only pages should yield NoChunkableContentError."""
    chunker = RecursiveChunker(settings=chunk_settings)

    with pytest.raises(NoChunkableContentError):
        chunker.chunk([_page("   "), _page("\n\t")])


def test_metadata_preservation(chunk_settings: Settings) -> None:
    """Loader metadata must survive chunking; chunk fields must be added."""
    text = "Paragraph one. " * 20
    chunker = RecursiveChunker(settings=chunk_settings)

    chunks = chunker.chunk([_page(text, source="/data/a.pdf", page=3, file_name="a.pdf")])

    assert len(chunks) >= 1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata[MetadataKeys.SOURCE] == "/data/a.pdf"
        assert chunk.metadata[MetadataKeys.PAGE] == 3
        assert chunk.metadata[MetadataKeys.FILE_NAME] == "a.pdf"
        assert chunk.metadata[MetadataKeys.CHUNK_INDEX] == index
        assert chunk.metadata[MetadataKeys.TOTAL_CHUNKS] == len(chunks)
        assert chunk.metadata[MetadataKeys.CHUNK_SIZE] == len(chunk.page_content)
        assert MetadataKeys.CHUNK_ID in chunk.metadata


def test_overlap_correctness() -> None:
    """Consecutive chunks should share overlapping characters when overlap > 0."""
    settings = Settings(chunk_size=20, chunk_overlap=5, separators=[""])
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    chunker = RecursiveChunker(settings=settings)

    chunks = chunker.chunk([_page(text)])

    assert len(chunks) >= 2
    first = chunks[0].page_content
    second = chunks[1].page_content
    overlap = settings.chunk_overlap
    assert first[-overlap:] == second[:overlap]


def test_invalid_settings_rejected_by_pydantic() -> None:
    """Settings validation must reject invalid chunk_size / overlap combinations."""
    with pytest.raises(ValidationError):
        Settings(chunk_size=0)

    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=150)

    with pytest.raises(ValidationError):
        Settings(chunk_overlap=-1)

    with pytest.raises(ValidationError):
        Settings(separators=[])


def test_chunk_count_with_known_input() -> None:
    """One long page should produce multiple chunks under a small chunk_size."""
    text = "ABCDEFGHIJ" * 30  # 300 chars
    settings = Settings(chunk_size=100, chunk_overlap=0, separators=[""])
    chunker = RecursiveChunker(settings=settings)

    chunks = chunker.chunk([_page(text)])

    assert len(chunks) == 3
    assert chunks[0].metadata[MetadataKeys.TOTAL_CHUNKS] == 3


def test_deterministic_output(chunk_settings: Settings) -> None:
    """Same documents + settings must produce identical chunks (regression-friendly)."""
    text = "Deterministic chunking matters for eval baselines. " * 15
    docs = [_page(text)]
    chunker = RecursiveChunker(settings=chunk_settings)

    first = chunker.chunk(docs)
    second = chunker.chunk(docs)

    assert [c.page_content for c in first] == [c.page_content for c in second]
    assert [c.metadata[MetadataKeys.CHUNK_ID] for c in first] == [
        c.metadata[MetadataKeys.CHUNK_ID] for c in second
    ]


def test_one_page_to_multiple_chunks_example() -> None:
    """README scenario: 1 page can become multiple chunks (e.g. ~5)."""
    settings = Settings(chunk_size=40, chunk_overlap=5, separators=[" ", ""])
    page_text = " ".join(f"token{i}" for i in range(80))
    chunker = RecursiveChunker(settings=settings)

    chunks = chunker.chunk([_page(page_text)])

    assert len(chunks) >= 5
    assert chunks[0].metadata[MetadataKeys.TOTAL_CHUNKS] == len(chunks)
