"""Domain exceptions for the embedding stage."""


class EmbeddingError(Exception):
    """Base error for all embedding failures."""


class EmptyDocumentListError(EmbeddingError):
    """Raised when ``embed_documents`` is called with an empty list."""


class EmptyQueryError(EmbeddingError):
    """Raised when ``embed_query`` receives blank text."""


class InvalidEmbeddingConfigError(EmbeddingError):
    """Raised when embedding settings are invalid at runtime."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the underlying embedding model cannot be loaded."""


class UnsupportedEmbeddingModelError(EmbeddingError):
    """Raised when the configured model id is empty or clearly unsupported."""


class EmbeddingInferenceError(EmbeddingError):
    """Raised when encoding / inference fails for documents or a query."""
