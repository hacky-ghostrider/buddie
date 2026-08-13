"""Tests for the embedding engine (mocked model — no torch load required)."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.config.settings import Settings
from app.embeddings.exceptions import (
    EmptyDocumentListError,
    EmptyQueryError,
    EmbeddingModelLoadError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.models import EmbeddedDocument
from app.embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from app.ingestion.metadata_keys import MetadataKeys


EMBED_DIM = 384


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _fake_encode(
    texts: list[str],
    *,
    normalize_embeddings: bool = True,
    **_kwargs: Any,
) -> list[list[float]]:
    """Deterministic fake encoder: hash characters into a fixed-dim vector."""
    vectors: list[list[float]] = []
    for text in texts:
        base = float(sum(ord(ch) for ch in text) % 97) + 1.0
        vec = [(base + i) * 0.01 for i in range(EMBED_DIM)]
        if normalize_embeddings:
            vec = _normalize(vec)
        vectors.append(vec)
    return vectors


@pytest.fixture
def embedding_settings() -> Settings:
    """Settings with a small batch size to exercise batching paths."""
    return Settings(
        app_env="test",
        embedding_model="BAAI/bge-small-en-v1.5",
        embed_batch_size=2,
        normalize_embeddings=True,
    )


@pytest.fixture
def fake_model() -> MagicMock:
    """Mock embedding model that never downloads weights or loads torch."""
    model = MagicMock()
    model.get_sentence_embedding_dimension.return_value = EMBED_DIM
    model.encode.side_effect = _fake_encode
    return model


@pytest.fixture
def embedder(
    embedding_settings: Settings,
    fake_model: MagicMock,
) -> SentenceTransformerEmbedding:
    """Embedding Strategy wired to the fake model."""
    return SentenceTransformerEmbedding(
        settings=embedding_settings,
        model=fake_model,
    )


def _doc(text: str, **meta: object) -> Document:
    metadata = {
        MetadataKeys.SOURCE: "/data/a.pdf",
        MetadataKeys.PAGE: 0,
        MetadataKeys.FILE_NAME: "a.pdf",
        MetadataKeys.CHUNK_ID: "a::p0::c0",
        MetadataKeys.CHUNK_INDEX: 0,
        **meta,
    }
    return Document(page_content=text, metadata=metadata)


def test_model_loads_via_factory(embedding_settings: Settings, fake_model: MagicMock) -> None:
    """Factory path should load the configured model name once."""
    calls: list[str] = []

    def factory(name: str) -> MagicMock:
        calls.append(name)
        return fake_model

    embedder = SentenceTransformerEmbedding(
        settings=embedding_settings,
        model_factory=factory,
    )

    _ = embedder.dimension
    assert calls == ["BAAI/bge-small-en-v1.5"]
    assert embedder.model_name == "BAAI/bge-small-en-v1.5"


def test_embedding_dimension(embedder: SentenceTransformerEmbedding) -> None:
    """Vectors must match the model dimension."""
    results = embedder.embed_documents([_doc("hello world")])

    assert embedder.dimension == EMBED_DIM
    assert results[0].dimension == EMBED_DIM
    assert len(results[0].embedding) == EMBED_DIM


def test_metadata_preserved(embedder: SentenceTransformerEmbedding) -> None:
    """Ingestion metadata must survive embedding unchanged."""
    results = embedder.embed_documents(
        [_doc("alpha", **{MetadataKeys.CHUNK_INDEX: 7, MetadataKeys.TOTAL_CHUNKS: 9})]
    )

    meta = results[0].metadata
    assert meta[MetadataKeys.SOURCE] == "/data/a.pdf"
    assert meta[MetadataKeys.FILE_NAME] == "a.pdf"
    assert meta[MetadataKeys.CHUNK_INDEX] == 7
    assert meta[MetadataKeys.TOTAL_CHUNKS] == 9
    assert results[0].text == "alpha"


def test_deterministic_output(embedder: SentenceTransformerEmbedding) -> None:
    """Same texts + settings must produce identical vectors (regression-friendly)."""
    docs = [_doc("deterministic embedding text"), _doc("second chunk")]

    first = embedder.embed_documents(docs)
    second = embedder.embed_documents(docs)

    assert [item.embedding for item in first] == [item.embedding for item in second]
    assert [item.text for item in first] == [item.text for item in second]


def test_batching_invokes_encode_per_batch(
    embedding_settings: Settings,
    fake_model: MagicMock,
) -> None:
    """With batch_size=2 and 5 docs, encode should run multiple batches."""
    embedder = SentenceTransformerEmbedding(settings=embedding_settings, model=fake_model)
    docs = [_doc(f"text-{i}") for i in range(5)]

    results = embedder.embed_documents(docs)

    assert len(results) == 5
    assert fake_model.encode.call_count == 3


def test_empty_document_list_raises(embedder: SentenceTransformerEmbedding) -> None:
    """Empty input must raise EmptyDocumentListError."""
    with pytest.raises(EmptyDocumentListError):
        embedder.embed_documents([])


def test_blank_documents_only_raises(embedder: SentenceTransformerEmbedding) -> None:
    """Whitespace-only documents are not embeddable."""
    with pytest.raises(EmptyDocumentListError):
        embedder.embed_documents([_doc("   "), _doc("\n")])


def test_empty_query_raises(embedder: SentenceTransformerEmbedding) -> None:
    """Blank queries must raise EmptyQueryError."""
    with pytest.raises(EmptyQueryError):
        embedder.embed_query("  ")


def test_embed_query_returns_vector(embedder: SentenceTransformerEmbedding) -> None:
    """Query embedding should be a non-empty float vector of model dimension."""
    vector = embedder.embed_query("what is RAG?")
    assert len(vector) == EMBED_DIM
    assert all(isinstance(v, float) for v in vector)


def test_invalid_model_name_in_settings() -> None:
    """Blank EMBEDDING_MODEL must fail settings validation."""
    with pytest.raises(ValidationError):
        Settings(embedding_model="   ")


def test_invalid_batch_size_in_settings() -> None:
    """Non-positive EMBED_BATCH_SIZE must fail settings validation."""
    with pytest.raises(ValidationError):
        Settings(embed_batch_size=0)


def test_model_load_failure(embedding_settings: Settings) -> None:
    """Factory exceptions must map to EmbeddingModelLoadError."""

    def boom(_name: str) -> MagicMock:
        raise RuntimeError("weights missing")

    embedder = SentenceTransformerEmbedding(
        settings=embedding_settings,
        model_factory=boom,
    )

    with pytest.raises(EmbeddingModelLoadError):
        _ = embedder.dimension


def test_unsupported_model_at_runtime() -> None:
    """Empty model after bypassing pydantic should raise UnsupportedEmbeddingModelError."""
    settings = Settings(embedding_model="BAAI/bge-small-en-v1.5")
    object.__setattr__(settings, "embedding_model", "   ")

    with pytest.raises(UnsupportedEmbeddingModelError):
        SentenceTransformerEmbedding(settings=settings, model=MagicMock())


def test_embedded_document_rejects_empty_vector() -> None:
    """EmbeddedDocument validation should reject empty embeddings."""
    with pytest.raises(ValidationError):
        EmbeddedDocument(text="hi", embedding=[], metadata={})
