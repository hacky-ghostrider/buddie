"""Abstract retriever contract.

Selects the most relevant stored chunks for a natural-language query.
Independent of LLMs, prompt templates, and generation — retrieval only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.retrieval.models import RetrievedDocument


class Retriever(ABC):
    """Strategy interface for query → top-k document retrieval."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the most relevant documents for ``query``.

        Args:
            query: User or evaluation question text.
            top_k: Optional override for how many neighbors to return.
                When ``None``, use configured ``TOP_K``.
            score_threshold: Optional minimum similarity score (inclusive).
                When ``None``, use configured ``DEFAULT_SCORE_THRESHOLD``.
            metadata_filters: Optional exact-match metadata constraints
                forwarded to the vector store. ``None`` means no filter.

        Returns:
            Retrieved documents ordered by descending similarity score.
            May be empty when nothing meets the threshold.

        Raises:
            EmptyQueryError: Query is blank.
            InvalidTopKError: ``top_k`` is not positive.
            InvalidScoreThresholdError: Threshold outside ``[0, 1]``.
            EmptyVectorStoreError: Collection has no vectors.
            RetrievalSearchError: Embedding or search failed.
        """
