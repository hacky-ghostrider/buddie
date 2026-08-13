"""Retrieval stage — query embedding + similarity search → top-k chunks.

Dense retrieval only in Sprint 6. No LLM generation, prompts, or evaluation.
"""

from app.retrieval.base import Retriever
from app.retrieval.exceptions import (
    EmptyQueryError,
    EmptyVectorStoreError,
    InvalidScoreThresholdError,
    InvalidTopKError,
    RetrievalError,
    RetrievalSearchError,
)
from app.retrieval.models import RetrievedDocument
from app.retrieval.vector_retriever import VectorRetriever

__all__ = [
    "Retriever",
    "VectorRetriever",
    "RetrievedDocument",
    "RetrievalError",
    "EmptyQueryError",
    "EmptyVectorStoreError",
    "InvalidTopKError",
    "InvalidScoreThresholdError",
    "RetrievalSearchError",
]
