"""Document ingestion service — load then optionally chunk documents.

Composition:
- ``DocumentLoaderFactory`` selects a loader Strategy
- ``Chunker`` (optional) splits loaded pages into retrieval-sized chunks
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.ingestion.chunking.base import Chunker
from app.ingestion.loaders.factory import DocumentLoaderFactory

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Application service for loading (and optionally chunking) documents.

    Args:
        settings: Optional settings override.
        loader_factory: Optional factory (inject a fake registry in tests).
        chunker: Optional chunker Strategy. When provided, ``load_and_chunk``
            applies it after loading. ``load`` remains load-only.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        loader_factory: DocumentLoaderFactory | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._loader_factory = loader_factory or DocumentLoaderFactory(settings=self._settings)
        self._chunker = chunker

    def load(self, file_path: str | Path) -> list[Document]:
        """Load a document through the configured loader factory (no chunking).

        Args:
            file_path: Path to a local document file.

        Returns:
            LangChain documents produced by the resolved loader.

        Raises:
            UnsupportedDocumentTypeError: No loader for the extension.
            DocumentLoaderError: Propagated from the concrete loader.
        """
        path = Path(file_path)
        loader = self._loader_factory.create(path)
        logger.info(
            "Ingestion started: loader=%s path=%s",
            type(loader).__name__,
            path,
        )
        documents = loader.load()
        logger.info(
            "Ingestion finished: path=%s document_count=%s",
            path,
            len(documents),
        )
        return documents

    def load_and_chunk(self, file_path: str | Path) -> list[Document]:
        """Load a document and chunk it using the injected chunker.

        Args:
            file_path: Path to a local document file.

        Returns:
            Chunk documents with enriched metadata.

        Raises:
            RuntimeError: No chunker was injected into the service.
            DocumentLoaderError: Propagated from loading.
            ChunkingError: Propagated from chunking.
        """
        if self._chunker is None:
            raise RuntimeError(
                "No chunker configured. Pass a Chunker to DocumentIngestionService "
                "or call RecursiveChunker.chunk() directly."
            )
        documents = self.load(file_path)
        return self._chunker.chunk(documents)
