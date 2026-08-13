"""Recursive character chunker using an in-house recursive splitter.

Independent of PDF loading and embeddings: accepts ``list[Document]`` and
returns chunk ``Document`` objects with enriched metadata.

Uses ``RecursiveCharacterSplitter`` (local) instead of importing
``langchain_text_splitters`` at package root, which eagerly pulls
``sentence_transformers`` / ``torch``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.ingestion.chunking.base import Chunker
from app.ingestion.chunking.exceptions import (
    ChunkingError,
    EmptyDocumentListError,
    InvalidChunkConfigError,
    NoChunkableContentError,
)
from app.ingestion.chunking.metadata import attach_chunk_metadata
from app.ingestion.chunking.recursive_splitter import RecursiveCharacterSplitter

logger = logging.getLogger(__name__)

SplitterFactory = Callable[[], RecursiveCharacterSplitter]


class RecursiveChunker(Chunker):
    """Split documents using recursive character splitting with configured separators.

    Args:
        settings: Provides ``chunk_size``, ``chunk_overlap``, and ``separators``.
        splitter_factory: Optional factory for injecting a fake splitter in tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        splitter_factory: SplitterFactory | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._splitter_factory = splitter_factory or self._default_splitter_factory

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Chunk documents into overlapping text segments.

        Args:
            documents: Input documents from a loader (or any upstream stage).

        Returns:
            Chunk documents preserving parent metadata plus chunk fields
            (``chunk_id``, ``chunk_index``, ``total_chunks``, ``chunk_size``).

        Raises:
            EmptyDocumentListError: ``documents`` is empty.
            NoChunkableContentError: All page contents are blank.
            InvalidChunkConfigError: Settings violate size/overlap rules.
            ChunkingError: Unexpected splitter failure.
        """
        logger.info("Chunking started: document_count=%s", len(documents))

        if not documents:
            logger.error("Chunking failed: empty document list")
            raise EmptyDocumentListError("Cannot chunk an empty document list")

        usable = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        skipped = len(documents) - len(usable)
        if skipped:
            logger.warning(
                "Skipped documents with empty page content: skipped=%s kept=%s",
                skipped,
                len(usable),
            )
        if not usable:
            logger.error("Chunking failed: no chunkable content")
            raise NoChunkableContentError("No documents contain usable page content")

        logger.info("Documents received for chunking: count=%s", len(usable))
        self._assert_runtime_config()

        try:
            splitter = self._splitter_factory()
            raw_chunks = splitter.split_documents(usable)
        except (EmptyDocumentListError, NoChunkableContentError, InvalidChunkConfigError):
            raise
        except Exception as exc:  # noqa: BLE001 — map vendor failures to domain error
            logger.error("Chunking failure from splitter: error=%s", exc)
            raise ChunkingError(f"Unexpected chunking failure: {exc}") from exc

        if not raw_chunks:
            logger.error("Splitter produced zero chunks")
            raise NoChunkableContentError("Splitter produced no chunks from input documents")

        enriched = attach_chunk_metadata(raw_chunks)
        avg_size = sum(len(c.page_content) for c in enriched) / len(enriched)

        logger.info(
            "Chunks generated: count=%s average_chunk_size=%.1f configured_size=%s overlap=%s",
            len(enriched),
            avg_size,
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        return enriched

    def _assert_runtime_config(self) -> None:
        """Validate chunk settings before invoking the splitter."""
        chunk_size = self._settings.chunk_size
        overlap = self._settings.chunk_overlap
        separators = self._settings.separators

        if chunk_size <= 0:
            raise InvalidChunkConfigError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise InvalidChunkConfigError(f"chunk_overlap must be >= 0, got {overlap}")
        if overlap > chunk_size:
            raise InvalidChunkConfigError(
                f"chunk_overlap ({overlap}) cannot exceed chunk_size ({chunk_size})"
            )
        if not separators:
            raise InvalidChunkConfigError("separators must be a non-empty list")

    def _default_splitter_factory(self) -> RecursiveCharacterSplitter:
        """Build the recursive character splitter from settings."""
        return RecursiveCharacterSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=list(self._settings.separators),
            length_function=len,
        )


def create_recursive_chunker(
    settings: Settings | None = None,
    *,
    separators: Sequence[str] | None = None,
) -> RecursiveChunker:
    """Convenience factory for ``RecursiveChunker``.

    Args:
        settings: Optional settings instance.
        separators: Optional override; when provided, a shallow settings copy
            is not applied — prefer configuring via ``Settings`` in production.

    Returns:
        Configured ``RecursiveChunker``.
    """
    cfg = settings or get_settings()
    if separators is not None:
        # Tests may pass explicit separators without mutating global settings.
        cfg = cfg.model_copy(update={"separators": list(separators)})
    return RecursiveChunker(settings=cfg)
