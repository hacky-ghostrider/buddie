"""Unit tests for demo Chroma corpus seeding (no live embeddings)."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.config.settings import Settings
from app.demo.seed_corpus import ensure_demo_corpus
from app.embeddings.models import EmbeddedDocument
from app.ingestion.metadata_keys import MetadataKeys
from app.vectorstore.models import AddDocumentsResult


def _settings(tmp_path) -> Settings:
    return Settings(
        vector_db="chroma",
        chroma_collection_name="rag_documents",
        chroma_persist_directory=str(tmp_path / "chroma"),
        chunk_size=200,
        chunk_overlap=20,
    )


def test_ensure_demo_corpus_skips_when_populated(tmp_path) -> None:
    store = MagicMock()
    store.collection_name = "rag_documents"
    store.collection_exists.return_value = True
    store.count.return_value = 3

    count = ensure_demo_corpus(_settings(tmp_path), vector_store=store)
    assert count == 3
    store.create_collection.assert_not_called()
    store.add_documents.assert_not_called()


def test_ensure_demo_corpus_creates_and_indexes(tmp_path) -> None:
    store = MagicMock()
    store.collection_name = "rag_documents"
    store.collection_exists.return_value = False
    store.count.side_effect = [2]
    store.add_documents.return_value = AddDocumentsResult(
        ids=["a", "b"],
        added_count=2,
        collection_name="rag_documents",
    )

    embedder = MagicMock()
    embedder.embed_documents.side_effect = lambda docs: [
        EmbeddedDocument(
            text=doc.page_content,
            embedding=[0.1, 0.2, 0.3],
            metadata=dict(doc.metadata),
        )
        for doc in docs
    ]

    count = ensure_demo_corpus(
        _settings(tmp_path),
        vector_store=store,
        embedding_model=embedder,
    )

    store.create_collection.assert_called_once()
    embedder.embed_documents.assert_called_once()
    store.add_documents.assert_called_once()
    assert count == 2
    embedded_batch = store.add_documents.call_args.args[0]
    assert all(MetadataKeys.CHUNK_ID in doc.metadata for doc in embedded_batch)
