"""Vector store package — persist embeddings independent of embedding models and retrievers."""

from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateDocumentIdError,
    EmptyDocumentListError,
    InvalidEmbeddingDimensionError,
    InvalidVectorStoreConfigError,
    MissingDocumentIdError,
    VectorStoreError,
    VectorStorePersistenceError,
)
from app.vectorstore.models import AddDocumentsResult, VectorDocument

__all__ = [
    "VectorStore",
    "ChromaVectorStore",
    "VectorDocument",
    "AddDocumentsResult",
    "VectorStoreError",
    "CollectionAlreadyExistsError",
    "CollectionNotFoundError",
    "DuplicateDocumentIdError",
    "MissingDocumentIdError",
    "InvalidEmbeddingDimensionError",
    "EmptyDocumentListError",
    "InvalidVectorStoreConfigError",
    "VectorStorePersistenceError",
]
