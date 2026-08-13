"""Domain exceptions for the vector store stage."""


class VectorStoreError(Exception):
    """Base error for all vector store failures."""


class CollectionAlreadyExistsError(VectorStoreError):
    """Raised when creating a collection that already exists."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when operating on a collection that does not exist."""


class DuplicateDocumentIdError(VectorStoreError):
    """Raised when adding a document whose id is already stored."""


class MissingDocumentIdError(VectorStoreError):
    """Raised when an embedded document has no resolvable id."""


class InvalidEmbeddingDimensionError(VectorStoreError):
    """Raised when embedding vectors have inconsistent or invalid dimensions."""


class EmptyDocumentListError(VectorStoreError):
    """Raised when ``add_documents`` / ``delete_documents`` receives an empty list."""


class InvalidVectorStoreConfigError(VectorStoreError):
    """Raised when vector store settings are invalid at runtime."""


class VectorStorePersistenceError(VectorStoreError):
    """Raised when disk persistence or client I/O fails."""
