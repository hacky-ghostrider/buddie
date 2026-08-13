"""Abstract vector store contract.

Persists embeddings and runs similarity search without depending on
embedding-model vendors, retrievers' orchestration, LLMs, or evaluation.
Callers pass already-embedded documents for writes and query vectors for search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.embeddings.models import EmbeddedDocument
from app.retrieval.models import RetrievedDocument
from app.vectorstore.models import AddDocumentsResult


class VectorStore(ABC):
    """Strategy interface for vector persistence and ANN search backends."""

    @abstractmethod
    def create_collection(self) -> None:
        """Create the configured collection if it does not already exist.

        Raises:
            CollectionAlreadyExistsError: Collection name is already taken.
            VectorStorePersistenceError: Underlying storage could not be created.
            InvalidVectorStoreConfigError: Collection name / path is invalid.
        """

    @abstractmethod
    def add_documents(self, documents: list[EmbeddedDocument]) -> AddDocumentsResult:
        """Persist embedded documents into the collection.

        Args:
            documents: Documents with text, embedding, and metadata.
                Each document must resolve to a stable id (see implementation).

        Returns:
            Ids and counts for the successfully stored documents.

        Raises:
            EmptyDocumentListError: No documents provided.
            CollectionNotFoundError: Target collection does not exist.
            MissingDocumentIdError: A document has no resolvable id.
            DuplicateDocumentIdError: An id already exists in the collection.
            InvalidEmbeddingDimensionError: Vector lengths are inconsistent.
            VectorStorePersistenceError: Disk / backend write failed.
        """

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        score_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Find nearest stored vectors to ``query_embedding``.

        Args:
            query_embedding: Dense query vector (same space as stored docs).
            top_k: Maximum number of neighbors to return (before/after
                threshold filtering — implementations return at most ``top_k``
                results that also satisfy ``score_threshold``).
            score_threshold: Minimum similarity score in ``[0, 1]`` (inclusive).
                Higher is more similar. Cosine-space stores convert distance to
                score as ``1 - distance``.
            metadata_filter: Optional exact-match metadata constraints
                (backend-specific ``where`` clause). ``None`` means no filter.

        Returns:
            Matches ordered by descending similarity score (best first).

        Raises:
            CollectionNotFoundError: Target collection does not exist.
            InvalidEmbeddingDimensionError: Query vector length is invalid.
            InvalidVectorStoreConfigError: ``top_k`` / threshold invalid.
            VectorStorePersistenceError: Backend query failed.
        """

    @abstractmethod
    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id from the collection.

        Args:
            ids: Document ids to remove.

        Raises:
            EmptyDocumentListError: Empty id list.
            CollectionNotFoundError: Target collection does not exist.
            VectorStorePersistenceError: Backend delete failed.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire configured collection.

        Raises:
            CollectionNotFoundError: Collection does not exist.
            VectorStorePersistenceError: Backend delete failed.
        """

    @abstractmethod
    def collection_exists(self) -> bool:
        """Return whether the configured collection currently exists.

        Returns:
            ``True`` if the collection is present, otherwise ``False``.
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors currently stored in the collection.

        Returns:
            Non-negative document count.

        Raises:
            CollectionNotFoundError: Collection does not exist.
            VectorStorePersistenceError: Backend count failed.
        """
