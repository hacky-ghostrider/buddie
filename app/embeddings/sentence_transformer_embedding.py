"""Sentence-Transformers embedding Strategy.

Uses a local ``SentenceTransformer`` model configured via settings.
Independent of vector stores and LLMs — only maps text → vectors.

``sentence_transformers`` / ``torch`` are imported lazily so unit tests can
inject a fake model without loading native DLLs at import time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from langchain_core.documents import Document

from app.config.settings import Settings, get_settings
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

logger = logging.getLogger(__name__)


class _SentenceTransformerLike(Protocol):
    """Minimal protocol used by this Strategy (real ST or test double)."""

    def get_sentence_embedding_dimension(self) -> int: ...

    def encode(self, *args: Any, **kwargs: Any) -> Any: ...


ModelFactory = Callable[[str], _SentenceTransformerLike]


def _default_model_factory(model_name: str) -> _SentenceTransformerLike:
    """Import and construct a real SentenceTransformer (heavy dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingModelLoadError(
            f"sentence-transformers is unavailable: {exc}"
        ) from exc
    return SentenceTransformer(model_name)


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embed documents and queries with a Sentence-Transformers model.

    Args:
        settings: Provides model name, batch size, and normalize flag.
        model_factory: Optional factory for constructing the model.
            Inject a fake in unit tests.
        model: Optional pre-built model instance (skips factory / lazy load).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        model_factory: ModelFactory | None = None,
        model: _SentenceTransformerLike | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_factory = model_factory or _default_model_factory
        self._model = model
        self._dimension: int | None = None
        self._validate_config()

    @property
    def model_name(self) -> str:
        """Return the configured embedding model id."""
        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality, loading the model if needed.

        Returns:
            Length of vectors produced by this model.
        """
        if self._dimension is None:
            model = self._get_model()
            self._dimension = int(model.get_sentence_embedding_dimension())
        return self._dimension

    def embed_documents(self, documents: list[Document]) -> list[EmbeddedDocument]:
        """Embed documents in batches and preserve metadata.

        Args:
            documents: Input documents/chunks to encode.

        Returns:
            ``EmbeddedDocument`` list aligned 1:1 with usable input texts.

        Raises:
            EmptyDocumentListError: No documents provided.
            EmbeddingInferenceError: Model encode failed.
        """
        if not documents:
            logger.error("embed_documents called with empty document list")
            raise EmptyDocumentListError("Cannot embed an empty document list")

        usable = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        skipped = len(documents) - len(usable)
        if skipped:
            logger.warning(
                "Skipped blank documents before embedding: skipped=%s kept=%s",
                skipped,
                len(usable),
            )
        if not usable:
            raise EmptyDocumentListError("No documents contain usable text to embed")

        texts = [doc.page_content for doc in usable]
        vectors = self._encode_texts(texts)

        embedded: list[EmbeddedDocument] = []
        for document, vector in zip(usable, vectors, strict=True):
            embedded.append(
                EmbeddedDocument(
                    text=document.page_content,
                    embedding=vector,
                    metadata=dict(document.metadata),
                )
            )

        logger.info(
            "Documents processed: count=%s embedding_dimension=%s batch_size=%s model=%s",
            len(embedded),
            self.dimension,
            self._settings.embed_batch_size,
            self.model_name,
        )
        return embedded

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Query text.

        Returns:
            Dense query embedding vector.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Cannot embed an empty query")

        vectors = self._encode_texts([query])
        logger.info(
            "Query embedded: model=%s dimension=%s",
            self.model_name,
            len(vectors[0]),
        )
        return vectors[0]

    def _validate_config(self) -> None:
        """Validate embedding-related settings before use."""
        model_name = self._settings.embedding_model.strip()
        if not model_name:
            raise UnsupportedEmbeddingModelError("EMBEDDING_MODEL must be a non-empty string")
        if self._settings.embed_batch_size <= 0:
            raise InvalidEmbeddingConfigError(
                f"EMBED_BATCH_SIZE must be positive, got {self._settings.embed_batch_size}"
            )

    def _get_model(self) -> _SentenceTransformerLike:
        """Lazy-load and cache the embedding model."""
        if self._model is not None:
            return self._model

        model_name = self.model_name
        logger.info("Model loading started: model=%s", model_name)
        try:
            self._model = self._model_factory(model_name)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001 — map vendor load errors
            logger.error("Model loading failed: model=%s error=%s", model_name, exc)
            raise EmbeddingModelLoadError(
                f"Failed to load embedding model '{model_name}': {exc}"
            ) from exc

        self._dimension = int(self._model.get_sentence_embedding_dimension())
        logger.info(
            "Model loading finished: model=%s embedding_dimension=%s",
            model_name,
            self._dimension,
        )
        return self._model

    def _encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts in configurable batches.

        Args:
            texts: Non-empty strings to embed.

        Returns:
            List of embedding vectors (Python floats).
        """
        model = self._get_model()
        batch_size = self._settings.embed_batch_size
        normalize = self._settings.normalize_embeddings
        logger.info(
            "Encoding started: text_count=%s batch_size=%s normalize=%s",
            len(texts),
            batch_size,
            normalize,
        )

        all_vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = model.encode(
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_vectors.extend(_to_float_vectors(encoded))
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference failure: error=%s", exc)
            raise EmbeddingInferenceError(f"Embedding inference failed: {exc}") from exc

        return all_vectors


def _to_float_vectors(encoded: Any) -> list[list[float]]:
    """Convert encode() output into a list of Python float vectors."""
    if hasattr(encoded, "tolist"):
        as_list = encoded.tolist()
    else:
        as_list = list(encoded)

    vectors: list[list[float]] = []
    for row in as_list:
        if hasattr(row, "tolist"):
            row = row.tolist()
        vectors.append([float(value) for value in row])
    return vectors
