"""Vector store factory — prefer Chroma, fall back on unsafe Windows hosts."""

from __future__ import annotations

import logging
import sys

from app.config.settings import Settings, get_settings
from app.vectorstore.base import VectorStore
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.exceptions import InvalidVectorStoreConfigError
from app.vectorstore.json_file_store import JsonFileVectorStore

logger = logging.getLogger(__name__)


def _hnswlib_available() -> bool:
    try:
        import hnswlib  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    """Construct the vector-store Strategy for this host / config.

    On Windows, Chroma ``collection.add`` often access-violates unless
    SegmentAPI + ``hnswlib`` are available. When that path is unsafe, use
    ``JsonFileVectorStore`` so RAG chat remains usable for local demos.

    Args:
        settings: Optional settings override.

    Returns:
        A ``VectorStore`` implementation.
    """
    cfg = settings or get_settings()
    backend = cfg.vector_db.strip().lower()

    if backend == "json":
        logger.info("Using JsonFileVectorStore (VECTOR_DB=json)")
        return JsonFileVectorStore(settings=cfg)

    if backend != "chroma":
        raise InvalidVectorStoreConfigError(
            f"VECTOR_DB must be 'chroma' or 'json', got '{cfg.vector_db}'"
        )

    if sys.platform == "win32" and not _hnswlib_available():
        logger.warning(
            "Windows Chroma add is unsafe without hnswlib; "
            "using JsonFileVectorStore fallback (collection=%s)",
            cfg.chroma_collection_name,
        )
        return JsonFileVectorStore(settings=cfg)

    logger.info("Using ChromaVectorStore: collection=%s", cfg.chroma_collection_name)
    return ChromaVectorStore(settings=cfg)
