"""EvaluationContext — single immutable snapshot for evaluation pipelines.

WHY (maintainability)
---------------------
Sprint 9–10 APIs passed many parallel parameters (question, answer, docs,
tokens, LangSmith ids, tool calls, …). Every new agent field meant another
kwarg on ``evaluate`` / automation helpers and easy signature drift.

``EvaluationContext`` is the **DTO / aggregate root** for one evaluation:
metrics, tool validators, tracers, and future LangGraph runners all read
from one object. Adding Sprint 11 fields becomes a model change, not a
signature rewrite across the codebase.

Java SDET analogy: prefer a typed ``TestContext`` / ``ScenarioContext``
over 15 method parameters on every page-object helper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.evaluation.tool_validation.tool_execution import ToolExecution
from app.evaluation.timeline import EvaluationTimeline
from app.orchestration.models import RAGResponse
from app.retrieval.models import RetrievedDocument


class EvaluationContext(BaseModel):
    """Everything required to evaluate one question / agent turn.

    Metric-facing code continues to use ``question``, ``answer``,
    ``retrieved_documents``, ``expected_answer``, ``expected_sources``,
    and ``metadata`` (Sprint 9 contract). Sprint 10.2 adds agent-ready
    observability fields without breaking those readers. Sprint 12 adds
    ``timeline`` for continuous-evaluation stage reconstruction.

    Attributes:
        question: Evaluated question text.
        original_user_request: Raw user utterance when it differs from
            the normalized question (multi-turn / agent planners).
        retrieved_documents: Structured retrieval evidence.
        retrieved_chunks: Plain chunk texts (trace / judge friendly).
        prompt: Prompt payload sent to the model.
        prompt_version: Template / prompt registry version string.
        tool_calls: Observed tool executions (empty until Sprint 11).
        tool_results: Tool outputs aligned with ``tool_calls`` when not
            already embedded on each ``ToolExecution.output``.
        answer: Generated answer (Sprint 9 field name; see
            ``generated_answer``).
        expected_answer: Optional golden answer.
        expected_sources: Optional expected source identifiers.
        model: LLM model id.
        latency_ms: End-to-end or stage latency in milliseconds.
        token_usage: Token accounting dictionary.
        cost_usd: Optional estimated USD cost.
        metadata: Free-form extras (suite name, tags, …).
        langsmith_trace_id: LangSmith trace id when traced.
        langsmith_run_id: LangSmith run id when traced.
        langsmith_run_url: Deep link into the LangSmith UI.
        correlation_id: Request / pipeline correlation id.
        timestamp: UTC snapshot time.
        timeline: Optional Sprint 12 evaluation timeline.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="Question under evaluation")
    original_user_request: str | None = Field(
        default=None,
        description="Original user request when distinct from question",
    )
    retrieved_documents: list[RetrievedDocument] = Field(
        default_factory=list,
        description="Retrieved evidence documents",
    )
    retrieved_chunks: list[str] = Field(
        default_factory=list,
        description="Retrieved chunk texts",
    )
    prompt: dict[str, Any] | str | None = Field(
        default=None,
        description="Prompt payload sent to the model",
    )
    prompt_version: str | None = Field(
        default=None,
        description="Prompt template / registry version",
    )
    tool_calls: list[ToolExecution] = Field(
        default_factory=list,
        description="Observed tool executions",
    )
    tool_results: list[Any] = Field(
        default_factory=list,
        description="Tool result payloads (optional companion list)",
    )
    answer: str = Field(
        default="",
        description="Generated answer text (Sprint 9 metric field name)",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Optional golden / expected answer",
    )
    expected_sources: list[str] | None = Field(
        default=None,
        description="Optional expected source identifiers",
    )
    model: str | None = Field(default=None, description="Model identifier")
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Observed latency in milliseconds",
    )
    token_usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage accounting",
    )
    cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional estimated cost in USD",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional evaluation metadata",
    )
    langsmith_trace_id: str | None = Field(
        default=None,
        description="LangSmith trace id",
    )
    langsmith_run_id: str | None = Field(
        default=None,
        description="LangSmith run id",
    )
    langsmith_run_url: str | None = Field(
        default=None,
        description="LangSmith run URL",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Correlation id for the evaluated request",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC snapshot timestamp",
    )
    timeline: EvaluationTimeline | None = Field(
        default=None,
        description="Optional continuous-evaluation stage timeline (Sprint 12)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def generated_answer(self) -> str:
        """Canonical name for the generated answer (alias of ``answer``)."""
        return self.answer

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject blank questions."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("EvaluationContext.question must be non-empty")
        return cleaned

    @classmethod
    def from_rag_response(
        cls,
        *,
        question: str,
        rag_response: RAGResponse,
        expected_answer: str | None = None,
        expected_sources: list[str] | None = None,
        tool_calls: list[ToolExecution] | None = None,
        tool_results: list[Any] | None = None,
        prompt_version: str | None = None,
        cost_usd: float | None = None,
        langsmith_trace_id: str | None = None,
        langsmith_run_id: str | None = None,
        langsmith_run_url: str | None = None,
        metadata: dict[str, Any] | None = None,
        original_user_request: str | None = None,
    ) -> EvaluationContext:
        """Build a context from a ``RAGResponse`` plus optional agent/trace fields.

        Args:
            question: Question under evaluation.
            rag_response: Orchestration output.
            expected_answer: Optional golden answer.
            expected_sources: Optional expected sources.
            tool_calls: Optional tool executions (Sprint 11 fills these).
            tool_results: Optional companion tool outputs.
            prompt_version: Optional prompt version string.
            cost_usd: Optional cost estimate.
            langsmith_trace_id: Optional LangSmith trace id.
            langsmith_run_id: Optional LangSmith run id.
            langsmith_run_url: Optional LangSmith URL.
            metadata: Optional extras.
            original_user_request: Optional raw user request.

        Returns:
            Populated ``EvaluationContext``.
        """
        gen_meta = dict(rag_response.generation_metadata or {})
        prompt = gen_meta.get("prompt")
        model = gen_meta.get("model")
        token_usage = {
            key: gen_meta[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in gen_meta
        }
        documents = list(rag_response.retrieved_documents)
        chunks = [doc.text for doc in documents]
        return cls(
            question=question,
            original_user_request=original_user_request or question,
            retrieved_documents=documents,
            retrieved_chunks=chunks,
            prompt=prompt,
            prompt_version=prompt_version or gen_meta.get("prompt_version"),
            tool_calls=list(tool_calls or []),
            tool_results=list(tool_results or []),
            answer=rag_response.answer,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            model=str(model) if model is not None else None,
            latency_ms=rag_response.latency.total_ms,
            token_usage=token_usage,
            cost_usd=cost_usd,
            metadata=dict(metadata or {}),
            langsmith_trace_id=langsmith_trace_id,
            langsmith_run_id=langsmith_run_id,
            langsmith_run_url=langsmith_run_url,
            correlation_id=rag_response.correlation_id,
        )


__all__ = ["EvaluationContext"]
