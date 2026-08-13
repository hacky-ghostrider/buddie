"""Custom exceptions for the document loading stage."""


class DocumentLoaderError(Exception):
    """Base error for all document loading failures."""


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when the target file path does not exist."""


class UnsupportedDocumentTypeError(DocumentLoaderError):
    """Raised when the file extension is not in the allowed set."""


class EmptyDocumentError(DocumentLoaderError):
    """Raised when a document loads successfully but contains no usable pages."""


class CorruptedDocumentError(DocumentLoaderError):
    """Raised when a document cannot be parsed (corrupt or invalid structure)."""


class DocumentPermissionError(DocumentLoaderError):
    """Raised when the process lacks permission to read the file."""
