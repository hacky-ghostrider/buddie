"""Canonical metadata keys for LangChain ``Document.metadata``.

Shared by loaders and chunkers so provenance fields stay consistent across
the ingestion pipeline.
"""

from __future__ import annotations

from enum import StrEnum


class MetadataKeys(StrEnum):
    """Typed keys written into ``Document.metadata`` during ingestion.

    Values match string keys expected by LangChain and by downstream
    RAG evaluation metrics (citation, groundedness, filtering).
    """

    # Loader / source provenance (Sprint 2)
    SOURCE = "source"
    PAGE = "page"
    FILE_NAME = "file_name"

    # Chunk provenance (Sprint 3)
    CHUNK_ID = "chunk_id"
    CHUNK_INDEX = "chunk_index"
    TOTAL_CHUNKS = "total_chunks"
    CHUNK_SIZE = "chunk_size"
