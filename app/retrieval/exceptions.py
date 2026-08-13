"""Domain exceptions for the retrieval stage."""


class RetrievalError(Exception):
    """Base error for all retrieval failures."""


class EmptyQueryError(RetrievalError):
    """Raised when ``retrieve`` receives blank query text."""


class InvalidTopKError(RetrievalError):
    """Raised when ``top_k`` is not a positive integer."""


class InvalidScoreThresholdError(RetrievalError):
    """Raised when ``score_threshold`` is outside the valid ``[0, 1]`` range."""


class EmptyVectorStoreError(RetrievalError):
    """Raised when similarity search is attempted against an empty collection."""


class RetrievalSearchError(RetrievalError):
    """Raised when embedding or vector search fails during retrieval."""
