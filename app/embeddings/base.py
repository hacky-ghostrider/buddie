"""Abstract embedding model contract.

Converts text into dense vectors without depending on vector databases,
retrievers, LLMs, or evaluation logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document

from app.embeddings.models import EmbeddedDocument


class EmbeddingModel(ABC):
    """Strategy interface for document and query embedding."""

    @abstractmethod
    def embed_documents(self, documents: list[Document]) -> list[EmbeddedDocument]:
        """Embed a batch of documents into vectors.

        Args:
            documents: LangChain documents (typically chunks) to encode.

        Returns:
            One ``EmbeddedDocument`` per input document, preserving metadata.

        Raises:
            EmbeddingError: On empty input, load failure, or inference failure.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a vector.

        Args:
            query: User or evaluation query text.

        Returns:
            Dense embedding vector for the query (same space as documents).

        Raises:
            EmbeddingError: On blank query, load failure, or inference failure.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the output embedding dimensionality for the loaded model.

        Returns:
            Vector length (e.g. 384 for ``BAAI/bge-small-en-v1.5``).
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model identifier.

        Returns:
            Hugging Face / sentence-transformers model id string.
        """
