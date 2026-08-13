"""Document loaders — Strategy-based ingestion for multiple sources."""

from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.exceptions import (
    CorruptedDocumentError,
    DocumentLoaderError,
    DocumentNotFoundError,
    DocumentPermissionError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
)
from app.ingestion.loaders.factory import DEFAULT_LOADER_REGISTRY, DocumentLoaderFactory
from app.ingestion.loaders.pdf_loader import PDFDocumentLoader
from app.ingestion.metadata_keys import MetadataKeys

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
