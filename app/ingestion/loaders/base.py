"""Abstract document loader contract.

Defines the Strategy interface every concrete loader must implement.
Callers (services, APIs, evaluation jobs) depend on this ABC — never on
PDF- or vendor-specific APIs — so new sources plug in without rewiring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class DocumentLoader(ABC):
    """Abstract base class for loading documents into LangChain ``Document`` objects.

    Concrete loaders encapsulate source-specific validation and parsing while
    returning a uniform ``list[Document]`` for downstream RAG stages.
    """

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return documents from the configured source.

        Returns:
            A list of LangChain ``Document`` instances (typically one per page
            or logical unit). Must not perform chunking.

        Raises:
            DocumentLoaderError: On validation or parsing failures. Concrete
                loaders should raise specific subclasses.
        """

    @abstractmethod
    def source_path(self) -> Path:
        """Return the resolved path (or identifier) of the document source.

        Returns:
            Path representing the primary source location for logging/metadata.
        """
