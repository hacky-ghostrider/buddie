"""Seed a minimal demo corpus into Chroma when the collection is missing.

The Streamlit / FastAPI chatbot needs ``rag_documents`` to exist before
``POST /api/v1/rag/query``. Offline ``make demo`` mocks retrieval and does not
create the collection — this helper bridges that gap for live UI smoke tests.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.embeddings.factory import build_embedding_model
from app.ingestion.chunking.recursive_chunker import RecursiveChunker
from app.ingestion.metadata_keys import MetadataKeys
from app.vectorstore.base import VectorStore
from app.vectorstore.exceptions import CollectionAlreadyExistsError
from app.vectorstore.factory import build_vector_store

logger = logging.getLogger(__name__)

_SAMPLE_HANDBOOK = (
    Path(__file__).resolve().parents[2] / "data" / "sample" / "employee_handbook.md"
)


def _demo_documents() -> list[Document]:
    """Build seed documents from the sample handbook (or inline fallback)."""
    if _SAMPLE_HANDBOOK.is_file():
        text = _SAMPLE_HANDBOOK.read_text(encoding="utf-8")
        source = str(_SAMPLE_HANDBOOK)
        file_name = _SAMPLE_HANDBOOK.name
    else:
        text = (
            "Employees accrue paid leave per the handbook policy. "
            "Vacation requires manager approval and advance notice.\n\n"
            "---\n\n"
            "You are chatting with Buddie, the AI Employee Assistant. "
            "It answers questions about work, leave, benefits, and policies."
        )
        source = "employee_handbook.md"
        file_name = "employee_handbook.md"

    # Split on the identity section so identity questions retrieve a tighter chunk.
    parts = [part.strip() for part in text.split("\n---\n") if part.strip()]
    if len(parts) == 1:
        parts = [text.strip()]

    documents: list[Document] = []
    for index, part in enumerate(parts):
        documents.append(
            Document(
                page_content=part,
                metadata={
                    MetadataKeys.SOURCE: source,
                    MetadataKeys.FILE_NAME: file_name,
                    MetadataKeys.PAGE: index,
                },
            )
        )
    return documents


def ensure_demo_corpus(
    settings: Settings | None = None,
    *,
    vector_store: VectorStore | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> int:
    """Create ``rag_documents`` and index demo chunks if the store is empty.

    Idempotent: when the collection already has vectors, returns the count
    without re-embedding.

    Args:
        settings: Application settings (collection name / persist path).
        vector_store: Optional shared store (reuse the API composition-root
            instance to avoid a second Chroma client on Windows).
        embedding_model: Optional shared embedder.

    Returns:
        Number of vectors in the collection after ensure.
    """
    cfg = settings or get_settings()
    store = vector_store or build_vector_store(cfg)

    if store.collection_exists():
        count = store.count()
        if count > 0:
            logger.info(
                "Demo corpus already present: collection=%s vectors=%s",
                store.collection_name,
                count,
            )
            return count
    else:
        try:
            store.create_collection()
        except CollectionAlreadyExistsError:
            logger.info(
                "Collection appeared concurrently: name=%s",
                store.collection_name,
            )

    embedder = embedding_model or build_embedding_model(cfg)
    chunker = RecursiveChunker(settings=cfg)
    chunks = chunker.chunk(_demo_documents())
    embedded = embedder.embed_documents(chunks)
    result = store.add_documents(embedded)

    logger.info(
        "Demo corpus seeded: collection=%s added=%s",
        result.collection_name,
        result.added_count,
    )
    return store.count()
