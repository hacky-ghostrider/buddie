"""Embedding package — text → vector Strategies independent of vector stores."""

from app.embeddings.base import EmbeddingModel
from app.embeddings.exceptions import (
    EmptyDocumentListError,
    EmptyQueryError,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    InvalidEmbeddingConfigError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.models import EmbeddedDocument
from app.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding

__all__ = [
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "EmbeddedDocument",
    "EmbeddingError",
    "EmptyDocumentListError",
    "EmptyQueryError",
    "InvalidEmbeddingConfigError",
    "EmbeddingModelLoadError",
    "UnsupportedEmbeddingModelError",
    "EmbeddingInferenceError",
]
