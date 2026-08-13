"""PDF document loader backed by ``pypdf`` (PyPDFLoader-compatible output).

LangChain's ``PyPDFLoader`` wraps the same ``pypdf`` library, but importing
``langchain_community.document_loaders`` can pull ``langchain_text_splitters``
→ ``sentence_transformers`` → ``torch``. We load pages with ``pypdf`` directly
and return LangChain ``Document`` objects with the same shape
(``page_content`` + ``source`` / ``page`` metadata).

Inject ``pdf_loader_factory=PyPDFLoader`` (or a test double) when you explicitly
want the LangChain class.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from pypdf import PdfReader

from app.config.settings import Settings, get_settings
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.exceptions import (
    CorruptedDocumentError,
    DocumentNotFoundError,
    DocumentPermissionError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.metadata_keys import MetadataKeys

logger = logging.getLogger(__name__)


class _PdfLoaderLike(Protocol):
    def load(self) -> list[Document]: ...


PdfLoaderFactory = Callable[[str], _PdfLoaderLike]


class PypdfPageLoader:
    """Minimal PDF page loader producing LangChain ``Document`` pages.

    Args:
        file_path: Path to a PDF file.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path

    def load(self) -> list[Document]:
        """Extract text from each PDF page.

        Returns:
            One ``Document`` per page (may be empty list for zero-page PDFs).

        Raises:
            Exception: Propagated parse errors for the caller to map.
        """
        reader = PdfReader(self._file_path)
        documents: list[Document] = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": self._file_path, "page": index},
                )
            )
        return documents


def _default_pdf_loader_factory(file_path: str) -> _PdfLoaderLike:
    """Construct the default PDF page loader."""
    return PypdfPageLoader(file_path)


class PDFDocumentLoader(DocumentLoader):
    """Load a local PDF file into a list of LangChain ``Document`` pages.

    Args:
        file_path: Absolute or relative path to a ``.pdf`` file.
        settings: Optional settings override (supported extensions, etc.).
        pdf_loader_factory: Optional factory used to construct the PDF loader.
            Defaults to LangChain ``PyPDFLoader``. Inject a fake in unit tests.
    """

    def __init__(
        self,
        file_path: str | Path,
        settings: Settings | None = None,
        pdf_loader_factory: PdfLoaderFactory | None = None,
    ) -> None:
        self._file_path = Path(file_path)
        self._settings = settings or get_settings()
        self._pdf_loader_factory = pdf_loader_factory or _default_pdf_loader_factory

    def source_path(self) -> Path:
        """Return the PDF file path being loaded.

        Returns:
            Resolved ``Path`` to the configured PDF.
        """
        return self._file_path

    def load(self) -> list[Document]:
        """Validate and load PDF pages as LangChain documents.

        Returns:
            One ``Document`` per PDF page, with ``source``, ``page``, and
            ``file_name`` metadata. Page text is raw extraction — not chunked.

        Raises:
            DocumentNotFoundError: File does not exist.
            UnsupportedDocumentTypeError: Extension not allowed by settings.
            DocumentPermissionError: OS denies read access.
            EmptyDocumentError: PDF has no pages / no extractable content.
            CorruptedDocumentError: PDF cannot be parsed.
        """
        logger.info("Loading started: path=%s", self._file_path)

        self._validate_file()
        logger.info("File validated: path=%s", self._file_path)

        raw_documents = self._load_raw_pages()
        documents = self._enrich_metadata(raw_documents)

        if not documents or all(not doc.page_content.strip() for doc in documents):
            logger.error("Empty PDF: path=%s pages=%s", self._file_path, len(documents))
            raise EmptyDocumentError(f"PDF contains no extractable content: {self._file_path}")

        logger.info(
            "Pages loaded: path=%s total_pages=%s",
            self._file_path,
            len(documents),
        )
        return documents

    def _validate_file(self) -> None:
        """Ensure the path exists, is readable, and has an allowed extension."""
        if not self._file_path.exists():
            logger.error("File not found: path=%s", self._file_path)
            raise DocumentNotFoundError(f"Document not found: {self._file_path}")

        if not self._file_path.is_file():
            logger.error("Path is not a file: path=%s", self._file_path)
            raise DocumentNotFoundError(f"Path is not a file: {self._file_path}")

        extension = self._file_path.suffix.lower()
        allowed = self._settings.get_supported_extensions()
        if extension not in allowed:
            logger.error(
                "Unsupported extension: path=%s extension=%s allowed=%s",
                self._file_path,
                extension,
                sorted(allowed),
            )
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension}' for {self._file_path}. "
                f"Allowed: {sorted(allowed)}"
            )

        try:
            with self._file_path.open("rb"):
                pass
        except PermissionError as exc:
            logger.error("Permission denied: path=%s", self._file_path)
            raise DocumentPermissionError(
                f"Permission denied reading document: {self._file_path}"
            ) from exc

    def _load_raw_pages(self) -> list[Document]:
        """Invoke PyPDFLoader and translate parse failures to domain errors."""
        try:
            loader = self._pdf_loader_factory(str(self._file_path))
            return loader.load()
        except DocumentPermissionError:
            raise
        except PermissionError as exc:
            logger.error("Permission denied while loading: path=%s", self._file_path)
            raise DocumentPermissionError(
                f"Permission denied reading document: {self._file_path}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — map vendor errors to domain type
            logger.error(
                "Failed to parse PDF: path=%s error=%s",
                self._file_path,
                exc,
            )
            raise CorruptedDocumentError(
                f"Failed to parse PDF (corrupt or invalid): {self._file_path}"
            ) from exc

    def _enrich_metadata(self, documents: list[Document]) -> list[Document]:
        """Normalize metadata required for RAG evaluation traceability.

        Args:
            documents: Raw documents from PyPDFLoader.

        Returns:
            Documents with ``source``, ``page``, and ``file_name`` set.
        """
        file_name = self._file_path.name
        source = str(self._file_path.resolve())
        enriched: list[Document] = []

        for index, document in enumerate(documents):
            metadata = dict(document.metadata)
            metadata[MetadataKeys.SOURCE] = metadata.get(MetadataKeys.SOURCE, source)
            metadata[MetadataKeys.FILE_NAME] = file_name
            if MetadataKeys.PAGE not in metadata:
                metadata[MetadataKeys.PAGE] = index
            enriched.append(
                Document(page_content=document.page_content, metadata=metadata)
            )

        return enriched
