"""Pure-Python hashing embedding Strategy (no torch / sentence-transformers).

Used as a Windows / offline fallback when native embedding DLLs fail to load.
Documents and queries must share this Strategy so they land in the same space.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

from langchain_core.documents import Document

from app.embeddings.base import EmbeddingModel
from app.embeddings.exceptions import (
    EmptyDocumentListError,
    EmptyQueryError,
    InvalidEmbeddingConfigError,
)
from app.embeddings.models import EmbeddedDocument

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DEFAULT_DIMENSION = 384  # Match BAAI/bge-small-en-v1.5 width for store compatibility
_MODEL_NAME = "hashing-fallback-v1"


class HashingEmbeddingModel(EmbeddingModel):
    """Bag-of-tokens hashing trick with L2-normalized vectors.

    Args:
        dimension: Output vector length (must be positive).
        model_name: Identifier reported in logs / metadata.
    """

    def __init__(
        self,
        dimension: int = _DEFAULT_DIMENSION,
        model_name: str = _MODEL_NAME,
    ) -> None:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise InvalidEmbeddingConfigError(
                f"dimension must be a positive integer, got {dimension!r}"
            )
        cleaned_name = model_name.strip()
        if not cleaned_name:
            raise InvalidEmbeddingConfigError("model_name must be a non-empty string")
        self._dimension = dimension
        self._model_name = cleaned_name
        logger.info(
            "HashingEmbeddingModel ready: dimension=%s model=%s",
            self._dimension,
            self._model_name,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, documents: list[Document]) -> list[EmbeddedDocument]:
        if not documents:
            raise EmptyDocumentListError("Cannot embed an empty document list")
        results: list[EmbeddedDocument] = []
        for document in documents:
            text = document.page_content if document.page_content is not None else ""
            if not text.strip():
                raise EmptyDocumentListError("Cannot embed a document with blank text")
            results.append(
                EmbeddedDocument(
                    text=text,
                    embedding=self._embed_text(text),
                    metadata=dict(document.metadata or {}),
                )
            )
        return results

    def embed_query(self, query: str) -> list[float]:
        cleaned = query.strip() if query else ""
        if not cleaned:
            raise EmptyQueryError("Query must be a non-empty string")
        return self._embed_text(cleaned)

    def _embed_text(self, text: str) -> list[float]:
        """Hash tokens into a fixed-size sparse-ish dense vector and L2-normalize."""
        vec = [0.0] * self._dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            # Stable non-zero vector for punctuation-only input.
            tokens = ["_empty_"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Use first 8 bytes for index / sign.
            value = int.from_bytes(digest[:8], "big")
            index = value % self._dimension
            sign = 1.0 if (value // self._dimension) % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(component * component for component in vec)) or 1.0
        return [component / norm for component in vec]
