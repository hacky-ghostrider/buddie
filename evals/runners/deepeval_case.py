"""DeepEval-compatible evaluation case (Sprint 18 phase 1 foundation).

Holds the four fields DeepEval ``LLMTestCase`` needs for upcoming metrics.
Does not invoke DeepEval metrics yet.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DeepEvalCompatibleCase(BaseModel):
    """One executed golden case mapped to DeepEval ``LLMTestCase`` fields.

    Mapping:
        input ← user_query
        actual_output ← runtime final answer
        expected_output ← expected_answer
        retrieval_context ← runtime RAG chunks and/or tool evidence texts only
        expected_output ← golden expected_answer (reference; never as evidence)
        expected_context ← golden reference context (never copied into retrieval)
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str
    input: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str] = Field(
        default_factory=list,
        description="Actual runtime RAG/tool evidence only; empty when none",
    )
    context: list[str] = Field(
        default_factory=list,
        description="Alias of retrieval_context for HallucinationMetric-style APIs",
    )
    category: str | None = None
    expected_behavior: str | None = None
    expected_context: list[str] = Field(
        default_factory=list,
        description=(
            "Golden reference context only — never substituted into "
            "retrieval_context"
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id", "input", "actual_output", "expected_output")
    @classmethod
    def _text_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("DeepEval case text fields must be non-empty")
        return cleaned

    def model_post_init(self, __context: Any) -> None:
        """Keep ``context`` aligned with ``retrieval_context`` when omitted."""
        if not self.context and self.retrieval_context:
            self.context = list(self.retrieval_context)

    def to_llm_test_case_kwargs(self) -> dict[str, Any]:
        """Keyword args suitable for ``deepeval.test_case.LLMTestCase``."""
        texts = list(self.retrieval_context)
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "retrieval_context": texts,
            "context": list(self.context) if self.context else texts,
        }


__all__ = ["DeepEvalCompatibleCase"]
