"""Tests for the retrieval layer.

Uses the disk-backed Chroma client double from vector-store tests so ANN
search is exercised without fragile native bindings. EmbeddingModel is a
deterministic fake so retrieval is reproducible for evaluation regression.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.embeddings.models import EmbeddedDocument
from app.ingestion.metadata_keys import MetadataKeys
from app.retrieval.exceptions import (
    EmptyQueryError,
    EmptyVectorStoreError,
    InvalidScoreThresholdError,
    InvalidTopKError,
)
from app.retrieval.models import RetrievedDocument
from app.retrieval.vector_retriever import VectorRetriever
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.exceptions import CollectionNotFoundError
from tests.test_vectorstore import _fake_client_factory


EMBED_DIM = 8


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _unit_axis(index: int, dim: int = EMBED_DIM) -> list[float]:
    """Return a one-hot-ish unit vector along ``index`` (deterministic)."""
    vec = [0.0] * dim
    vec[index % dim] = 1.0
    return vec


@pytest.fixture
def retrieval_settings(tmp_path: Path) -> Settings:
    """Isolated store + retrieval defaults for tests."""
    return Settings(
        app_env="test",
        vector_db="chroma",
        chroma_collection_name="rag_retrieval_docs",
        chroma_persist_directory=str(tmp_path / "chroma"),
        top_k=3,
        default_score_threshold=0.0,
        normalize_embeddings=True,
    )


@pytest.fixture
def store(retrieval_settings: Settings) -> ChromaVectorStore:
    """ChromaVectorStore with query-capable fake client."""
    return ChromaVectorStore(
        settings=retrieval_settings,
        client_factory=_fake_client_factory,
    )


def _embedded(
    text: str,
    *,
    doc_id: str,
    embedding: list[float],
    **meta: object,
) -> EmbeddedDocument:
    metadata = {
        MetadataKeys.SOURCE: "/data/a.pdf",
        MetadataKeys.PAGE: 0,
        MetadataKeys.FILE_NAME: "a.pdf",
        MetadataKeys.CHUNK_ID: doc_id,
        MetadataKeys.CHUNK_INDEX: 0,
        **meta,
    }
    return EmbeddedDocument(
        text=text,
        embedding=_normalize(list(embedding)),
        metadata=metadata,
    )


def _seed_corpus(store: ChromaVectorStore) -> None:
    """Seed three orthogonal-ish chunks for predictable ranking."""
    store.create_collection()
    store.add_documents(
        [
            _embedded(
                "cats and kittens prefer warm blankets",
                doc_id="cat-1",
                embedding=_unit_axis(0),
                topic="animals",
            ),
            _embedded(
                "dogs and puppies enjoy outdoor walks",
                doc_id="dog-1",
                embedding=_unit_axis(1),
                topic="animals",
            ),
            _embedded(
                "quantum computing uses qubits and superposition",
                doc_id="qc-1",
                embedding=_unit_axis(2),
                topic="science",
            ),
        ]
    )


def _fake_embedding_model(query_to_vector: dict[str, list[float]] | None = None) -> MagicMock:
    """EmbeddingModel double: maps known queries to fixed vectors."""
    mapping = query_to_vector or {}
    model = MagicMock()
    model.model_name = "fake-embedder"

    def _embed_query(query: str) -> list[float]:
        cleaned = query.strip()
        if cleaned in mapping:
            return _normalize(list(mapping[cleaned]))
        # Default: lean toward axis 0 ("cats") for unknown queries.
        return _normalize(_unit_axis(0))

    model.embed_query.side_effect = _embed_query
    return model


@pytest.fixture
def retriever(store: ChromaVectorStore, retrieval_settings: Settings) -> VectorRetriever:
    """VectorRetriever with seeded corpus and deterministic embedder."""
    _seed_corpus(store)
    embedding_model = _fake_embedding_model(
        {
            "tell me about cats": _unit_axis(0),
            "tell me about dogs": _unit_axis(1),
            "what is quantum computing": _unit_axis(2),
            "unrelated astronomy topic": _unit_axis(7),
        }
    )
    return VectorRetriever(
        embedding_model=embedding_model,
        vector_store=store,
        settings=retrieval_settings,
    )


def test_retrieve_top_k(retriever: VectorRetriever) -> None:
    """retrieve should honor top_k and return at most that many docs."""
    results = retriever.retrieve("tell me about cats", top_k=2)
    assert len(results) == 2
    assert results[0].id == "cat-1"


def test_empty_query(retriever: VectorRetriever) -> None:
    """Blank queries must raise EmptyQueryError."""
    with pytest.raises(EmptyQueryError):
        retriever.retrieve("   ")


def test_metadata_preservation(retriever: VectorRetriever) -> None:
    """Retrieved metadata must preserve ingestion fields."""
    results = retriever.retrieve("tell me about cats", top_k=1)
    assert results[0].metadata[MetadataKeys.CHUNK_ID] == "cat-1"
    assert results[0].metadata[MetadataKeys.FILE_NAME] == "a.pdf"
    assert results[0].metadata["topic"] == "animals"


def test_score_ordering(retriever: VectorRetriever) -> None:
    """Results must be ordered by descending similarity score."""
    results = retriever.retrieve("tell me about dogs", top_k=3)
    assert len(results) >= 2
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].id == "dog-1"


def test_threshold_filtering(
    store: ChromaVectorStore,
    retrieval_settings: Settings,
) -> None:
    """High thresholds should drop weak matches."""
    _seed_corpus(store)
    embedding_model = _fake_embedding_model(
        {"tell me about cats": _unit_axis(0)}
    )
    retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=store,
        settings=retrieval_settings,
    )
    strict = retriever.retrieve(
        "tell me about cats",
        top_k=5,
        score_threshold=0.99,
    )
    assert len(strict) == 1
    assert strict[0].id == "cat-1"

    loose = retriever.retrieve(
        "tell me about cats",
        top_k=5,
        score_threshold=0.0,
    )
    assert len(loose) >= 1


def test_no_results_after_threshold(
    store: ChromaVectorStore,
    retrieval_settings: Settings,
) -> None:
    """High thresholds with an off-axis query yield an empty list."""
    _seed_corpus(store)
    embedding_model = _fake_embedding_model(
        {"unrelated astronomy topic": _unit_axis(7)}
    )
    retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=store,
        settings=retrieval_settings,
    )
    results = retriever.retrieve(
        "unrelated astronomy topic",
        top_k=3,
        score_threshold=0.5,
    )
    assert results == []


def test_empty_vector_store(
    retrieval_settings: Settings,
    store: ChromaVectorStore,
) -> None:
    """Retrieving against an empty collection must raise EmptyVectorStoreError."""
    store.create_collection()
    retriever = VectorRetriever(
        embedding_model=_fake_embedding_model(),
        vector_store=store,
        settings=retrieval_settings,
    )
    with pytest.raises(EmptyVectorStoreError):
        retriever.retrieve("tell me about cats")


def test_missing_collection(
    retrieval_settings: Settings,
    store: ChromaVectorStore,
) -> None:
    """Missing collection bubbles CollectionNotFoundError."""
    retriever = VectorRetriever(
        embedding_model=_fake_embedding_model(),
        vector_store=store,
        settings=retrieval_settings,
    )
    with pytest.raises(CollectionNotFoundError):
        retriever.retrieve("tell me about cats")


def test_invalid_top_k(retriever: VectorRetriever) -> None:
    """Non-positive top_k must raise InvalidTopKError."""
    with pytest.raises(InvalidTopKError):
        retriever.retrieve("tell me about cats", top_k=0)


def test_invalid_score_threshold(retriever: VectorRetriever) -> None:
    """Out-of-range thresholds must raise InvalidScoreThresholdError."""
    with pytest.raises(InvalidScoreThresholdError):
        retriever.retrieve("tell me about cats", score_threshold=1.5)


def test_deterministic_retrieval(retriever: VectorRetriever) -> None:
    """Same query + fixed embedder must yield identical ids and scores.

    Deterministic retrieval matters for evaluation regression: if the same
    question suddenly ranks different chunks, you cannot tell whether the
    *retriever* or the *generator* regressed.
    """
    first = retriever.retrieve("what is quantum computing", top_k=2)
    second = retriever.retrieve("what is quantum computing", top_k=2)
    assert [item.id for item in first] == [item.id for item in second]
    assert [item.score for item in first] == [item.score for item in second]
    assert first[0].id == "qc-1"


def test_similarity_search_direct(store: ChromaVectorStore) -> None:
    """VectorStore.similarity_search returns scored RetrievedDocument rows."""
    _seed_corpus(store)
    hits = store.similarity_search(
        _normalize(_unit_axis(1)),
        top_k=2,
        score_threshold=0.0,
    )
    assert len(hits) == 2
    assert hits[0].id == "dog-1"
    assert isinstance(hits[0], RetrievedDocument)
    assert hits[0].score >= hits[1].score


def test_metadata_filter(store: ChromaVectorStore) -> None:
    """Optional metadata_filter restricts candidates (exact match)."""
    _seed_corpus(store)
    hits = store.similarity_search(
        _normalize(_unit_axis(0)),
        top_k=5,
        score_threshold=0.0,
        metadata_filter={"topic": "science"},
    )
    assert hits
    assert all(item.metadata.get("topic") == "science" for item in hits)


def test_settings_validate_top_k() -> None:
    """TOP_K must be positive."""
    with pytest.raises(ValidationError):
        Settings(top_k=0)


def test_settings_validate_score_threshold() -> None:
    """DEFAULT_SCORE_THRESHOLD must lie in [0, 1]."""
    with pytest.raises(ValidationError):
        Settings(default_score_threshold=1.1)


def test_retrieved_document_rejects_blank_id() -> None:
    """RetrievedDocument validation rejects blank ids."""
    with pytest.raises(ValidationError):
        RetrievedDocument(id="  ", text="hi", metadata={}, score=0.5)
