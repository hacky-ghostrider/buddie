"""Pydantic models for the vector store stage.

Why IDs matter
--------------
A vector database is an addressable store. Without stable document IDs you
cannot delete, update, or cite a specific chunk later. Analogy: a library
without call numbers — you can shelve books, but you cannot find or remove
a specific one reliably.

Why metadata stays separate from vectors
----------------------------------------
The embedding is geometry (where meaning sits). Metadata is provenance
(source file, page, chunk index). Mixing them into one blob makes filtering
and evaluation harder. Keep vectors for similarity; keep metadata for
filters, citations, and debugging.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VectorDocument(BaseModel):
    """One persisted unit in a vector collection.

    Attributes:
        id: Stable unique identifier (typically chunk_id from ingestion).
        text: Original text that was embedded (stored for later retrieval).
        embedding: Dense vector for this text.
        metadata: Provenance and filter fields (not part of the vector).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable unique document id")
    text: str = Field(description="Source text associated with the embedding")
    embedding: list[float] = Field(description="Dense embedding vector")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Separate provenance / filter fields",
    )

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ids — Chroma and most stores require non-empty ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("VectorDocument.id must be non-empty")
        return cleaned

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank text payloads."""
        if not value.strip():
            raise ValueError("VectorDocument.text must be non-empty")
        return value

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_non_empty(cls, value: list[float]) -> list[float]:
        """Reject empty vectors."""
        if not value:
            raise ValueError("VectorDocument.embedding must be a non-empty vector")
        return value

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""
        return len(self.embedding)


class AddDocumentsResult(BaseModel):
    """Outcome of a successful ``add_documents`` call.

    Attributes:
        ids: Document ids that were persisted.
        added_count: Number of vectors written.
        collection_name: Target collection.
    """

    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(description="Persisted document ids")
    added_count: int = Field(ge=0, description="Number of documents added")
    collection_name: str = Field(description="Collection that received the documents")
