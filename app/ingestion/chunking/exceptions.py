"""Domain exceptions for the chunking stage."""


class ChunkingError(Exception):
    """Base error for all chunking failures."""


class EmptyDocumentListError(ChunkingError):
    """Raised when the input document list is empty."""


class NoChunkableContentError(ChunkingError):
    """Raised when documents exist but none contain usable text."""


class InvalidChunkConfigError(ChunkingError):
    """Raised when chunk_size / overlap / separators are invalid at runtime."""
