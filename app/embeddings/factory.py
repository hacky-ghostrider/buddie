"""Embedding model factory — prefer Sentence-Transformers, fall back if DLLs fail."""

from __future__ import annotations

import logging

from app.config.settings import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.embeddings.exceptions import EmbeddingModelLoadError
from app.embeddings.hashing_embedding import HashingEmbeddingModel
from app.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding

logger = logging.getLogger(__name__)


def build_embedding_model(settings: Settings | None = None) -> EmbeddingModel:
    """Construct the best available ``EmbeddingModel`` for this host.

    Tries ``SentenceTransformerEmbedding`` (production path). When native
    ``torch`` / ``sentence-transformers`` DLLs fail (common on some Windows
    setups), falls back to ``HashingEmbeddingModel`` so RAG chat still works
    for demo / smoke tests.

    Args:
        settings: Optional settings override.

    Returns:
        A ready-to-use embedding Strategy.
    """
    cfg = settings or get_settings()
    try:
        model = SentenceTransformerEmbedding(settings=cfg)
        # Force native load now so failure is caught before indexing queries.
        _ = model.dimension
        logger.info(
            "Using SentenceTransformerEmbedding: model=%s dimension=%s",
            model.model_name,
            model.dimension,
        )
        return model
    except EmbeddingModelLoadError as exc:
        logger.warning(
            "Sentence-Transformers unavailable (%s); "
            "falling back to HashingEmbeddingModel",
            exc,
        )
        return HashingEmbeddingModel()
