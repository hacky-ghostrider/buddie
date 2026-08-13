"""Abstract chunker contract.

Transforms loaded documents into smaller ``Document`` chunks without
depending on PDF loaders, embeddings, or vector stores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class Chunker(ABC):
    """Strategy interface for splitting documents into retrieval-sized chunks."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split input documents into chunk documents with enriched metadata.

        Args:
            documents: Source documents (typically one per page from a loader).

        Returns:
            Chunk ``Document`` instances. Existing source metadata is preserved;
            chunk-specific metadata is added.

        Raises:
            ChunkingError: On invalid input or splitter failures.
        """
