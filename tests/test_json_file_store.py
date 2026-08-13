"""Unit tests for JsonFileVectorStore."""

from __future__ import annotations

from app.config.settings import Settings
from app.embeddings.models import EmbeddedDocument
from app.ingestion.metadata_keys import MetadataKeys
from app.vectorstore.json_file_store import JsonFileVectorStore


def _settings(tmp_path) -> Settings:
    return Settings(
        vector_db="json",
        chroma_collection_name="rag_documents",
        chroma_persist_directory=str(tmp_path / "vectors"),
    )


def _doc(text: str, doc_id: str, embedding: list[float]) -> EmbeddedDocument:
    return EmbeddedDocument(
        text=text,
        embedding=embedding,
        metadata={MetadataKeys.CHUNK_ID: doc_id},
    )


def test_json_file_store_roundtrip(tmp_path) -> None:
    store = JsonFileVectorStore(_settings(tmp_path))
    store.create_collection()
    store.add_documents(
        [
            _doc("leave policy vacation accrual", "h1", [1.0, 0.0, 0.0]),
            _doc("quantum computing qubits", "q1", [0.0, 1.0, 0.0]),
        ]
    )
    assert store.count() == 2
    hits = store.similarity_search([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].id == "h1"
    assert "leave" in hits[0].text
