"""Dense vector retriever Strategy.

Orchestrates: query text → EmbeddingModel.embed_query → VectorStore.similarity_search.

Document embeddings (ingestion) and query embeddings (retrieval) stay conceptually
separate even when they share one model: ingestion encodes corpus chunks once;
retrieval encodes a live question each request — different lifecycle, batching,
and failure modes. Keeping ``embed_documents`` / ``embed_query`` distinct also
allows asymmetric models later (e.g. query-tuned BGE variants).
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import Settings, get_settings
from app.embeddings.base import EmbeddingModel
from app.embeddings.exceptions import EmbeddingError
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
from app.vectorstore.base import VectorStore
from app.vectorstore.exceptions import (
    CollectionNotFoundError,
    InvalidEmbeddingDimensionError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


class VectorRetriever(Retriever):
    """Retrieve top-k chunks via query embedding + vector similarity search.

    Args:
        embedding_model: Strategy used to embed the query string.
        vector_store: Persistence / ANN search backend.
        settings: Provides default ``TOP_K`` and ``DEFAULT_SCORE_THRESHOLD``.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Embed ``query`` and return the best-matching stored chunks."""
        cleaned = query.strip() if query else ""
        if not cleaned:
            logger.error("Retrieval rejected: empty query")
            raise EmptyQueryError("Query must be a non-empty string")

        resolved_top_k = self._settings.top_k if top_k is None else top_k
        resolved_threshold = (
            self._settings.default_score_threshold
            if score_threshold is None
            else score_threshold
        )
        self._validate_top_k(resolved_top_k)
        self._validate_score_threshold(resolved_threshold)

        logger.info(
            "Retrieval started: query_preview=%r top_k=%s score_threshold=%s "
            "has_metadata_filters=%s",
            cleaned[:80],
            resolved_top_k,
            resolved_threshold,
            metadata_filters is not None,
        )

        try:
            vector_count = self._vector_store.count()
        except CollectionNotFoundError:
            logger.error("Retrieval failed: collection missing")
            raise
        except VectorStoreError as exc:
            logger.error("Retrieval failed while counting vectors: error=%s", exc)
            raise RetrievalSearchError(
                f"Failed to inspect vector store before search: {exc}"
            ) from exc

        if vector_count == 0:
            logger.error("Retrieval failed: vector store is empty")
            raise EmptyVectorStoreError(
                "Cannot retrieve from an empty vector collection"
            )

        try:
            query_embedding = self._embedding_model.embed_query(cleaned)
        except EmbeddingError as exc:
            logger.error("Query embedding failed: error=%s", exc)
            raise RetrievalSearchError(
                f"Failed to embed query: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Query embedding failed unexpectedly: error=%s", exc)
            raise RetrievalSearchError(
                f"Failed to embed query: {exc}"
            ) from exc

        logger.info(
            "Query embedding generated: dimension=%s model=%s",
            len(query_embedding),
            getattr(self._embedding_model, "model_name", type(self._embedding_model).__name__),
        )

        try:
            results = self._vector_store.similarity_search(
                query_embedding,
                top_k=resolved_top_k,
                score_threshold=resolved_threshold,
                metadata_filter=metadata_filters,
            )
        except (CollectionNotFoundError, InvalidEmbeddingDimensionError):
            raise
        except VectorStoreError as exc:
            logger.error("Similarity search failed: error=%s", exc)
            raise RetrievalSearchError(
                f"Similarity search failed: {exc}"
            ) from exc
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Similarity search failed unexpectedly: error=%s", exc)
            raise RetrievalSearchError(
                f"Similarity search failed: {exc}"
            ) from exc

        avg_score = (
            sum(item.score for item in results) / len(results) if results else 0.0
        )
        logger.info(
            "Retrieval finished: retrieved_count=%s average_score=%.4f",
            len(results),
            avg_score,
        )
        return results

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Reject non-positive top_k values."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidTopKError(f"top_k must be a positive integer, got {top_k!r}")

    @staticmethod
    def _validate_score_threshold(score_threshold: float) -> None:
        """Reject thresholds outside the cosine-normalized ``[0, 1]`` range."""
        try:
            value = float(score_threshold)
        except (TypeError, ValueError) as exc:
            raise InvalidScoreThresholdError(
                f"score_threshold must be a float in [0, 1], got {score_threshold!r}"
            ) from exc
        if value < 0.0 or value > 1.0:
            raise InvalidScoreThresholdError(
                f"score_threshold must be between 0 and 1 inclusive, got {value}"
            )
