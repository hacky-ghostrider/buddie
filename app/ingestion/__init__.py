"""Ingestion package — document loading and chunking pipeline stages.

Prefer importing concrete modules (e.g. ``app.ingestion.chunking``) to keep
import graphs light. This package uses lazy ``__getattr__`` exports.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DocumentLoader",
    "DocumentLoaderFactory",
    "PDFDocumentLoader",
    "MetadataKeys",
    "Chunker",
    "RecursiveChunker",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve public ingestion symbols."""
    if name == "MetadataKeys":
        from app.ingestion.metadata_keys import MetadataKeys

        return MetadataKeys
    if name in {"DocumentLoader", "DocumentLoaderFactory", "PDFDocumentLoader"}:
        from app.ingestion import loaders as loaders_pkg

        return getattr(loaders_pkg, name)
    if name in {"Chunker", "RecursiveChunker"}:
        if name == "Chunker":
            from app.ingestion.chunking.base import Chunker

            return Chunker
        from app.ingestion.chunking.recursive_chunker import RecursiveChunker

        return RecursiveChunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
