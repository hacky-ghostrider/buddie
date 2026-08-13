"""Pydantic models for the embedding stage.

``EmbeddedDocument`` binds text, vector, and provenance together so downstream
stages (vector store, retrieval, evaluation) never juggle parallel lists that
can drift out of sync.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddedDocument(BaseModel):
    """A single text unit paired with its embedding vector and metadata.

    Attributes:
        text: Original chunk/page text that was embedded.
        embedding: Dense vector representation (list of floats).
        metadata: Provenance and chunk fields preserved from ingestion.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Source text that was embedded")
    embedding: list[float] = Field(description="Dense embedding vector")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved metadata from loaders/chunkers",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank text — embeddings of empty strings are not useful."""
        if not value.strip():
            raise ValueError("EmbeddedDocument.text must be non-empty")
        return value

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_non_empty(cls, value: list[float]) -> list[float]:
        """Reject empty vectors."""
        if not value:
            raise ValueError("EmbeddedDocument.embedding must be a non-empty vector")
        return value

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality.

        Returns:
            Length of the embedding vector.
        """
        return len(self.embedding)
