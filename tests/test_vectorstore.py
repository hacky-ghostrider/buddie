"""Tests for the vector store layer.

Uses a disk-backed Chroma client double so tests exercise real persistence
semantics without depending on native Chroma bindings (known to access-violate
on some Windows hosts during ``collection.add``).

Why persistence tests matter: production processes restart. If vectors vanish
after reopen, retrieval and evaluation become nondeterministic ghosts.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.embeddings.models import EmbeddedDocument
from app.ingestion.metadata_keys import MetadataKeys
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateDocumentIdError,
    EmptyDocumentListError,
    InvalidEmbeddingDimensionError,
    MissingDocumentIdError,
)
from app.vectorstore.models import VectorDocument


EMBED_DIM = 8


class _FakeCollection:
    """Minimal Chroma-like collection backed by an in-memory dict."""

    def __init__(
        self,
        name: str,
        store: dict[str, dict[str, Any]],
        on_change: Any,
    ) -> None:
        self.name = name
        self._store = store
        self._on_change = on_change

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for doc_id, embedding, text, metadata in zip(
            ids, embeddings, documents, metadatas, strict=True
        ):
            if doc_id in self._store:
                raise ValueError(f"DuplicateIDError: id already exists: {doc_id}")
            self._store[doc_id] = {
                "embedding": list(embedding),
                "document": text,
                "metadata": dict(metadata),
            }
        self._on_change()

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Brute-force cosine-distance query (mirrors Chroma cosine space)."""
        del include  # always return full payload for the fake
        if not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query = list(query_embeddings[0])
        candidates: list[tuple[str, float, str, dict[str, Any]]] = []
        for doc_id, record in self._store.items():
            metadata = dict(record["metadata"])
            if where and not _metadata_matches(metadata, where):
                continue
            distance = _cosine_distance(query, list(record["embedding"]))
            candidates.append(
                (doc_id, distance, str(record["document"]), metadata)
            )

        candidates.sort(key=lambda item: item[1])
        top = candidates[: max(n_results, 0)]
        return {
            "ids": [[item[0] for item in top]],
            "documents": [[item[2] for item in top]],
            "metadatas": [[item[3] for item in top]],
            "distances": [[item[1] for item in top]],
        }

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._store.pop(doc_id, None)
        self._on_change()

    def count(self) -> int:
        return len(self._store)

    def get(
        self,
        ids: list[str] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        del include  # unused — API compatibility with Chroma
        if ids is None:
            selected = list(self._store.keys())
        else:
            selected = [doc_id for doc_id in ids if doc_id in self._store]
        return {
            "ids": selected,
            "documents": [self._store[i]["document"] for i in selected],
            "metadatas": [self._store[i]["metadata"] for i in selected],
            "embeddings": [self._store[i]["embedding"] for i in selected],
        }


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Return ``1 - cosine_similarity`` (Chroma cosine space distance)."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return 1.0 - similarity


def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Exact-match filter compatible with simple Chroma ``where`` dicts."""
    for key, expected in where.items():
        if metadata.get(key) != expected:
            return False
    return True


class _FakeChromaClient:
    """Disk-persisted fake Chroma client for contract + persistence tests."""

    def __init__(self, persist_directory: str) -> None:
        self._path = Path(persist_directory)
        self._path.mkdir(parents=True, exist_ok=True)
        self._state_file = self._path / "fake_chroma_state.json"
        self._collections: dict[str, dict[str, dict[str, Any]]] = self._load()

    def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self._state_file.exists():
            return {}
        return json.loads(self._state_file.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._state_file.write_text(
            json.dumps(self._collections, indent=2),
            encoding="utf-8",
        )

    def create_collection(
        self,
        name: str,
        *,
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None,
        get_or_create: bool = False,
    ) -> _FakeCollection:
        del embedding_function, metadata, get_or_create
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")
        self._collections[name] = {}
        self._save()
        return _FakeCollection(name, self._collections[name], self._save)

    def get_collection(
        self,
        name: str,
        *,
        embedding_function: Any = None,
    ) -> _FakeCollection:
        del embedding_function
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return _FakeCollection(name, self._collections[name], self._save)

    def delete_collection(self, name: str) -> None:
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        del self._collections[name]
        self._save()

    def list_collections(self) -> list[_FakeCollection]:
        return [
            _FakeCollection(name, docs, self._save)
            for name, docs in self._collections.items()
        ]


def _fake_client_factory(persist_directory: str) -> _FakeChromaClient:
    return _FakeChromaClient(persist_directory)


@pytest.fixture
def vector_settings(tmp_path: Path) -> Settings:
    """Settings pointing at an isolated temp persist directory."""
    return Settings(
        app_env="test",
        vector_db="chroma",
        chroma_collection_name="rag_test_docs",
        chroma_persist_directory=str(tmp_path / "chroma"),
    )


@pytest.fixture
def store(vector_settings: Settings) -> ChromaVectorStore:
    """ChromaVectorStore wired to the disk-backed fake client."""
    return ChromaVectorStore(
        settings=vector_settings,
        client_factory=_fake_client_factory,
    )


def _embedded(
    text: str,
    *,
    doc_id: str,
    dim: int = EMBED_DIM,
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
    embedding = [float((i + 1) * 0.01) for i in range(dim)]
    return EmbeddedDocument(text=text, embedding=embedding, metadata=metadata)


def test_collection_creation(store: ChromaVectorStore) -> None:
    """create_collection should make the collection visible."""
    assert store.collection_exists() is False
    store.create_collection()
    assert store.collection_exists() is True


def test_collection_already_exists(store: ChromaVectorStore) -> None:
    """Second create_collection must raise CollectionAlreadyExistsError."""
    store.create_collection()
    with pytest.raises(CollectionAlreadyExistsError):
        store.create_collection()


def test_add_documents(store: ChromaVectorStore) -> None:
    """Adding documents should persist ids and increase count."""
    store.create_collection()
    result = store.add_documents(
        [
            _embedded("alpha chunk", doc_id="a::p0::c0"),
            _embedded("beta chunk", doc_id="a::p0::c1"),
        ]
    )

    assert result.added_count == 2
    assert result.ids == ["a::p0::c0", "a::p0::c1"]
    assert result.collection_name == "rag_test_docs"
    assert store.count() == 2


def test_duplicate_ids(store: ChromaVectorStore) -> None:
    """Re-adding an existing id must raise DuplicateDocumentIdError."""
    store.create_collection()
    store.add_documents([_embedded("first", doc_id="dup-1")])

    with pytest.raises(DuplicateDocumentIdError):
        store.add_documents([_embedded("second", doc_id="dup-1")])


def test_duplicate_ids_in_batch(store: ChromaVectorStore) -> None:
    """Duplicate ids inside one batch must be rejected before write."""
    store.create_collection()
    with pytest.raises(DuplicateDocumentIdError):
        store.add_documents(
            [
                _embedded("one", doc_id="same"),
                _embedded("two", doc_id="same"),
            ]
        )


def test_delete_document(store: ChromaVectorStore) -> None:
    """delete_documents should remove targeted ids only."""
    store.create_collection()
    store.add_documents(
        [
            _embedded("keep me", doc_id="keep"),
            _embedded("drop me", doc_id="drop"),
        ]
    )

    store.delete_documents(["drop"])
    assert store.count() == 1


def test_delete_collection(store: ChromaVectorStore) -> None:
    """delete_collection removes the collection entirely."""
    store.create_collection()
    store.add_documents([_embedded("temp", doc_id="t1")])
    store.delete_collection()

    assert store.collection_exists() is False
    with pytest.raises(CollectionNotFoundError):
        store.count()


def test_count(store: ChromaVectorStore) -> None:
    """count reflects adds and deletes."""
    store.create_collection()
    assert store.count() == 0
    store.add_documents([_embedded("x", doc_id="x1"), _embedded("y", doc_id="y1")])
    assert store.count() == 2
    store.delete_documents(["x1"])
    assert store.count() == 1


def test_persistence_across_reopen(vector_settings: Settings) -> None:
    """Vectors must survive client reopen against the same directory.

    Persistence tests matter because deploy/restart must not lose the index.
    Analogy: saving a Word doc, closing Word, reopening — the text must still
    be there.
    """
    first = ChromaVectorStore(
        settings=vector_settings,
        client_factory=_fake_client_factory,
    )
    first.create_collection()
    first.add_documents(
        [
            _embedded("persisted one", doc_id="p1"),
            _embedded("persisted two", doc_id="p2"),
        ]
    )
    assert first.count() == 2

    second = ChromaVectorStore(
        settings=vector_settings,
        client_factory=_fake_client_factory,
    )
    assert second.collection_exists() is True
    assert second.count() == 2


def test_add_without_collection_raises(store: ChromaVectorStore) -> None:
    """Adding before create_collection must raise CollectionNotFoundError."""
    with pytest.raises(CollectionNotFoundError):
        store.add_documents([_embedded("orphan", doc_id="o1")])


def test_empty_add_raises(store: ChromaVectorStore) -> None:
    """Empty document list must raise EmptyDocumentListError."""
    store.create_collection()
    with pytest.raises(EmptyDocumentListError):
        store.add_documents([])


def test_missing_document_id_raises(store: ChromaVectorStore) -> None:
    """Documents without chunk_id/id metadata must raise MissingDocumentIdError."""
    store.create_collection()
    doc = EmbeddedDocument(
        text="no id here",
        embedding=[0.1] * EMBED_DIM,
        metadata={"source": "x.pdf"},
    )
    with pytest.raises(MissingDocumentIdError):
        store.add_documents([doc])


def test_invalid_embedding_dimensions(store: ChromaVectorStore) -> None:
    """Mixed dimensions in one batch must raise InvalidEmbeddingDimensionError."""
    store.create_collection()
    with pytest.raises(InvalidEmbeddingDimensionError):
        store.add_documents(
            [
                _embedded("dim8", doc_id="d8", dim=8),
                _embedded("dim4", doc_id="d4", dim=4),
            ]
        )


def test_vector_document_rejects_blank_id() -> None:
    """VectorDocument validation should reject blank ids."""
    with pytest.raises(ValidationError):
        VectorDocument(id="  ", text="hi", embedding=[0.1, 0.2], metadata={})


def test_settings_validate_vector_db() -> None:
    """Unsupported VECTOR_DB must fail settings validation."""
    with pytest.raises(ValidationError):
        Settings(vector_db="pinecone")


def test_settings_validate_collection_name() -> None:
    """Too-short CHROMA_COLLECTION_NAME must fail validation."""
    with pytest.raises(ValidationError):
        Settings(chroma_collection_name="ab")


def test_settings_validate_persist_directory() -> None:
    """Blank CHROMA_PERSIST_DIRECTORY must fail validation."""
    with pytest.raises(ValidationError):
        Settings(chroma_persist_directory="   ")


def _real_chroma_add_works(tmp_path: Path) -> bool:
    """Probe native Chroma add in a subprocess (avoids killing the test process)."""
    probe_dir = tmp_path / "real_chroma_probe"
    script = f"""
import sys
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
client = PersistentClient(
    path=r"{probe_dir}",
    settings=ChromaSettings(anonymized_telemetry=False),
)
col = client.create_collection(name="probe_col", embedding_function=None)
col.add(ids=["id1"], embeddings=[[0.1, 0.2, 0.3]], documents=["hello"], metadatas=[{{"k": "v"}}])
print(col.count())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.returncode == 0 and result.stdout.strip().endswith("1")


@pytest.mark.integration
def test_real_chroma_smoke_when_available(tmp_path: Path) -> None:
    """Optional smoke test against real Chroma when the native runtime works."""
    if not _real_chroma_add_works(tmp_path):
        pytest.skip(
            "Native Chroma collection.add is unsafe/unavailable on this host "
            "(common Windows Rust-binding access violation). "
            "Use Linux, Docker HttpClient, or SegmentAPI+hnswlib."
        )

    settings = Settings(
        app_env="test",
        vector_db="chroma",
        chroma_collection_name="real_smoke",
        chroma_persist_directory=str(tmp_path / "real_chroma_store"),
    )
    store = ChromaVectorStore(settings=settings)
    store.create_collection()
    result = store.add_documents([_embedded("real chroma", doc_id="real-1")])
    assert result.added_count == 1
    assert store.count() == 1
