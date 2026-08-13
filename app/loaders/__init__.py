"""Backward-compatible re-exports for Sprint 2 import paths.

Prefer ``app.ingestion.loaders`` going forward.
"""

from app.ingestion.loaders import (  # noqa: F401
    DEFAULT_LOADER_REGISTRY,
    CorruptedDocumentError,
    DocumentLoader,
    DocumentLoaderError,
    DocumentLoaderFactory,
    DocumentNotFoundError,
    DocumentPermissionError,
    EmptyDocumentError,
    MetadataKeys,
    PDFDocumentLoader,
    UnsupportedDocumentTypeError,
)

__all__ = [
    "DocumentLoader",
    "PDFDocumentLoader",
    "DocumentLoaderFactory",
    "DEFAULT_LOADER_REGISTRY",
    "MetadataKeys",
    "DocumentLoaderError",
    "DocumentNotFoundError",
    "UnsupportedDocumentTypeError",
    "EmptyDocumentError",
    "CorruptedDocumentError",
    "DocumentPermissionError",
]
