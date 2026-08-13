"""Pydantic models for the retrieval stage.

``RetrievedDocument`` is the atomic unit returned by search: text plus provenance
plus a similarity **score**. Scores matter for ranking, thresholding, and later
evaluation (hit-rate, MRR, nDCG) without re-running the embedder.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievedDocument(BaseModel):
    """One chunk returned by similarity search / retrieval.

    Attributes:
        id: Stable document id (typically ingestion ``chunk_id``).
        text: Stored chunk text (context for generation / evaluation).
        metadata: Provenance and filter fields from ingestion.
        score: Similarity score in ``[0, 1]`` (higher = more similar).
            Derived from cosine space as ``1 - distance`` when the store
            returns cosine distance.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable unique document id")
    text: str = Field(description="Retrieved chunk text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved provenance / filter fields",
    )
    score: float = Field(
        description="Similarity score (higher is better; typically cosine-derived)",
    )

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, value: str) -> str:
        """Reject blank ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("RetrievedDocument.id must be non-empty")
        return cleaned

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank retrieved text."""
        if not value.strip():
            raise ValueError("RetrievedDocument.text must be non-empty")
        return value
