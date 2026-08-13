"""Request / response contracts for the RAG orchestration stage.

``RAGRequest`` is the inbound API / service contract.
``RAGResponse`` is the outbound contract carrying answer, evidence, and
observability metadata (latencies, token estimates, usage).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.retrieval.models import RetrievedDocument


class LatencyBreakdown(BaseModel):
    """Per-stage wall-clock latencies for one RAG query.

    Attributes:
        retrieval_ms: Time spent in the Retriever.
        prompt_build_ms: Time spent building the prompt.
        llm_ms: Time spent in the LLM provider.
        total_ms: End-to-end orchestration latency.
    """

    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = Field(ge=0.0, description="Retriever duration in ms")
    prompt_build_ms: float = Field(ge=0.0, description="PromptBuilder duration in ms")
    llm_ms: float = Field(ge=0.0, description="LLMProvider duration in ms")
    total_ms: float = Field(ge=0.0, description="Total orchestration duration in ms")


class RAGRequest(BaseModel):
    """Inbound request for an end-to-end RAG query.

    Attributes:
        question: User / evaluation question.
        top_k: Optional override for retrieval depth.
        score_threshold: Optional override for minimum similarity.
        metadata_filters: Optional exact-match metadata constraints.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="User question to answer via RAG")
    top_k: int | None = Field(
        default=None,
        description="Optional retrieval top-k override (must be > 0 when set)",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Optional minimum cosine similarity in [0, 1]",
    )
    metadata_filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional exact-match metadata filters for retrieval",
    )

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive_when_set(cls, value: int | None) -> int | None:
        """Reject non-positive top_k overrides."""
        if value is None:
            return value
        if value <= 0:
            raise ValueError("top_k must be a positive integer when provided")
        return value

    @field_validator("score_threshold")
    @classmethod
    def score_threshold_must_be_unit_interval(
        cls,
        value: float | None,
    ) -> float | None:
        """Reject score thresholds outside ``[0, 1]``."""
        if value is None:
            return value
        if value < 0.0 or value > 1.0:
            raise ValueError(
                "score_threshold must be between 0 and 1 inclusive when provided"
            )
        return value


class RAGResponse(BaseModel):
    """Structured result of one end-to-end RAG query.

    Attributes:
        question: Echo of the cleaned question.
        answer: Generated answer text.
        retrieved_documents: Evidence chunks used for grounding.
        retrieval_metadata: Retrieval observability (counts, knobs, …).
        generation_metadata: Generation observability (model, tokens, latencies).
        latency: Per-stage and total latency breakdown.
        correlation_id: Request-scoped id for log correlation.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Cleaned question that was answered")
    answer: str = Field(description="Generated answer text")
    retrieved_documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Retrieved evidence documents",
    )
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieval-side observability fields",
    )
    generation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Generation-side observability fields (includes latencies)",
    )
    latency: LatencyBreakdown = Field(description="Per-stage latency breakdown")
    correlation_id: str = Field(description="Per-request correlation identifier")
