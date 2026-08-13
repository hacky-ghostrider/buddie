"""Tests for PDFDocumentLoader, factory registry, and ingestion service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.config.settings import Settings
from app.ingestion.loaders.exceptions import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    DocumentPermissionError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.loaders.factory import DocumentLoaderFactory
from app.ingestion.loaders.pdf_loader import PDFDocumentLoader
from app.ingestion.metadata_keys import MetadataKeys
from app.services.document_ingestion_service import DocumentIngestionService


def test_load_valid_pdf_returns_documents_with_metadata(
    valid_pdf: Path,
    settings: Settings,
) -> None:
    """A valid PDF should load at least one page with required metadata."""
    loader = PDFDocumentLoader(file_path=valid_pdf, settings=settings)

    documents = loader.load()

    assert len(documents) >= 1
    assert any(doc.page_content.strip() for doc in documents)
    first = documents[0]
    assert first.metadata[MetadataKeys.FILE_NAME] == valid_pdf.name
    assert MetadataKeys.SOURCE in first.metadata
    assert MetadataKeys.PAGE in first.metadata


def test_load_valid_pdf_with_mocked_pypdf_loader(
    valid_pdf: Path,
    settings: Settings,
    sample_documents: list[Document],
) -> None:
    """Mocked PyPDFLoader verifies enrichment without depending on PDF parsing."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = sample_documents

    loader = PDFDocumentLoader(
        file_path=valid_pdf,
        settings=settings,
        pdf_loader_factory=lambda _path: mock_loader,
    )

    documents = loader.load()

    assert len(documents) == 2
    assert documents[0].metadata[MetadataKeys.FILE_NAME] == "sample.pdf"
    assert documents[0].metadata[MetadataKeys.PAGE] == 0
    assert documents[1].metadata[MetadataKeys.PAGE] == 1
    mock_loader.load.assert_called_once()


def test_invalid_path_raises_not_found(settings: Settings, tmp_path: Path) -> None:
    """Missing files must raise DocumentNotFoundError."""
    missing = tmp_path / "does_not_exist.pdf"
    loader = PDFDocumentLoader(file_path=missing, settings=settings)

    with pytest.raises(DocumentNotFoundError):
        loader.load()


def test_unsupported_extension_raises(
    tmp_path: Path,
    settings: Settings,
) -> None:
    """Non-PDF extensions must raise UnsupportedDocumentTypeError."""
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")
    loader = PDFDocumentLoader(file_path=text_file, settings=settings)

    with pytest.raises(UnsupportedDocumentTypeError):
        loader.load()


def test_corrupted_pdf_raises(
    corrupted_pdf: Path,
    settings: Settings,
) -> None:
    """Unparseable PDF bytes must raise CorruptedDocumentError."""
    loader = PDFDocumentLoader(file_path=corrupted_pdf, settings=settings)

    with pytest.raises(CorruptedDocumentError):
        loader.load()


def test_empty_pdf_raises(empty_pdf: Path, settings: Settings) -> None:
    """A zero-page PDF must raise EmptyDocumentError."""
    loader = PDFDocumentLoader(file_path=empty_pdf, settings=settings)

    with pytest.raises(EmptyDocumentError):
        loader.load()


def test_empty_content_from_loader_raises(
    valid_pdf: Path,
    settings: Settings,
) -> None:
    """Pages with only whitespace are treated as empty documents."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="   ", metadata={"page": 0}),
    ]
    loader = PDFDocumentLoader(
        file_path=valid_pdf,
        settings=settings,
        pdf_loader_factory=lambda _path: mock_loader,
    )

    with pytest.raises(EmptyDocumentError):
        loader.load()


def test_permission_denied_raises(
    valid_pdf: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OS permission errors must map to DocumentPermissionError."""

    def _deny_open(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", _deny_open)
    loader = PDFDocumentLoader(file_path=valid_pdf, settings=settings)

    with pytest.raises(DocumentPermissionError):
        loader.load()


def test_settings_control_supported_extensions(tmp_path: Path) -> None:
    """Supported extensions come from settings, not hardcoded loader logic alone."""
    pdf = tmp_path / "blocked.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    restrictive = Settings(supported_document_types=".docx")

    loader = PDFDocumentLoader(file_path=pdf, settings=restrictive)

    with pytest.raises(UnsupportedDocumentTypeError):
        loader.load()


def test_factory_creates_pdf_loader(valid_pdf: Path, settings: Settings) -> None:
    """DocumentLoaderFactory should resolve PDFDocumentLoader for .pdf."""
    factory = DocumentLoaderFactory(settings=settings)
    loader = factory.create(valid_pdf)
    assert isinstance(loader, PDFDocumentLoader)


def test_factory_rejects_unknown_extension(tmp_path: Path, settings: Settings) -> None:
    """Factory must reject extensions outside settings."""
    html = tmp_path / "page.html"
    html.write_text("<html></html>", encoding="utf-8")
    factory = DocumentLoaderFactory(settings=settings)

    with pytest.raises(UnsupportedDocumentTypeError):
        factory.create(html)


def test_factory_registry_can_register_custom_loader(
    tmp_path: Path,
    settings: Settings,
) -> None:
    """Registry.register should allow extending supported types without if/else."""
    md_path = tmp_path / "notes.md"
    md_path.write_text("# hello", encoding="utf-8")

    class FakeMarkdownLoader(PDFDocumentLoader):
        """Minimal stub for registry extension tests."""

        def load(self) -> list[Document]:
            return [
                Document(
                    page_content="hello",
                    metadata={MetadataKeys.FILE_NAME: self.source_path().name},
                )
            ]

        def _validate_file(self) -> None:
            return None

    extended_settings = Settings(supported_document_types=".pdf,.md")
    factory = DocumentLoaderFactory(settings=extended_settings)
    factory.register(
        ".md",
        lambda path, cfg: FakeMarkdownLoader(file_path=path, settings=cfg),
    )

    loader = factory.create(md_path)
    docs = loader.load()
    assert docs[0].page_content == "hello"
    assert ".md" in factory.registered_extensions


def test_ingestion_service_delegates_to_factory(
    valid_pdf: Path,
    settings: Settings,
    sample_documents: list[Document],
) -> None:
    """Service should load via injected factory and not own creation logic."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = sample_documents

    factory = DocumentLoaderFactory(
        settings=settings,
        registry={
            ".pdf": lambda path, cfg: PDFDocumentLoader(
                file_path=path,
                settings=cfg,
                pdf_loader_factory=lambda _p: mock_loader,
            )
        },
    )
    service = DocumentIngestionService(settings=settings, loader_factory=factory)

    docs = service.load(valid_pdf)
    assert len(docs) == 2
    mock_loader.load.assert_called_once()


def test_ingestion_service_rejects_unknown_extension(
    tmp_path: Path,
    settings: Settings,
) -> None:
    """Service must surface UnsupportedDocumentTypeError from the factory."""
    html = tmp_path / "page.html"
    html.write_text("<html></html>", encoding="utf-8")
    service = DocumentIngestionService(settings=settings)

    with pytest.raises(UnsupportedDocumentTypeError):
        service.load(html)


def test_get_supported_extensions_normalizes_values() -> None:
    """Settings should normalize 'pdf' and '.PDF' to '.pdf'."""
    settings = Settings(supported_document_types="pdf, .DOCX")
    assert settings.get_supported_extensions() == frozenset({".pdf", ".docx"})


def test_metadata_keys_enum_values() -> None:
    """MetadataKeys values must stay stable for LangChain compatibility."""
    assert MetadataKeys.SOURCE == "source"
    assert MetadataKeys.PAGE == "page"
    assert MetadataKeys.FILE_NAME == "file_name"
    assert MetadataKeys.CHUNK_ID == "chunk_id"
    assert MetadataKeys.CHUNK_INDEX == "chunk_index"
    assert MetadataKeys.TOTAL_CHUNKS == "total_chunks"
    assert MetadataKeys.CHUNK_SIZE == "chunk_size"
