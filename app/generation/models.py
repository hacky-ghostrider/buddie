"""Pydantic models for the generation stage.

``BuiltPrompt`` is the contract between PromptBuilder and LLMProvider.
``GeneratedAnswer`` is the structured output of a single LLM call, including
token usage for cost, monitoring, and evaluation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TokenUsage(BaseModel):
    """Token accounting for one generation call.

    Attributes:
        prompt_tokens: Tokens consumed by the input prompt (system + user).
        completion_tokens: Tokens produced in the model completion.
        total_tokens: ``prompt_tokens + completion_tokens``.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0, description="Input / prompt token count")
    completion_tokens: int = Field(ge=0, description="Output / completion token count")
    total_tokens: int = Field(ge=0, description="Sum of prompt and completion tokens")

    @model_validator(mode="after")
    def total_must_match_parts(self) -> TokenUsage:
        """Ensure total equals the sum of prompt and completion tokens."""
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens + completion_tokens ({expected})"
            )
        return self


class BuiltPrompt(BaseModel):
    """Formatted prompt ready for an LLMProvider.

    Attributes:
        system: System instructions (role, grounding rules, style).
        user: User message containing context block + question.
        question: Original question text (for logging / evaluation).
        context_document_count: Number of retrieved docs injected.
        context_char_length: Approximate character size of injected context.
    """

    model_config = ConfigDict(extra="forbid")

    system: str = Field(description="System prompt text")
    user: str = Field(description="User prompt text (context + question)")
    question: str = Field(description="Original question for traceability")
    context_document_count: int = Field(
        ge=0,
        description="How many retrieved documents were injected",
    )
    context_char_length: int = Field(
        ge=0,
        description="Character length of the injected context block",
    )

    @field_validator("system", "user")
    @classmethod
    def prompts_must_not_be_blank(cls, value: str) -> str:
        """Reject blank system or user prompt bodies."""
        if not value.strip():
            raise ValueError("Prompt text must be non-empty")
        return value


class GeneratedAnswer(BaseModel):
    """Structured result of one LLM generation call.

    ``usage`` nests token counts; ``prompt_tokens`` / ``completion_tokens`` /
    ``total_tokens`` are also denormalized at the top level so callers and
    metrics scrapers can read them without digging into nested objects.

    Why usage matters
    -----------------
    - **Cost:** Providers bill per token; usage enables accurate chargeback.
    - **Monitoring:** Sudden usage spikes signal prompt bloat or loops.
    - **Evaluation:** Compare answer quality vs token budget across models.

    Attributes:
        answer: Model-generated answer text.
        model: Model id that produced the answer.
        usage: Nested token usage object.
        finish_reason: Provider stop reason (e.g. ``stop``, ``length``).
        prompt_tokens: Denormalized input token count.
        completion_tokens: Denormalized output token count.
        total_tokens: Denormalized total token count.
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Generated answer text")
    model: str = Field(description="Model identifier used for generation")
    usage: TokenUsage = Field(description="Nested token usage accounting")
    finish_reason: str | None = Field(
        default=None,
        description="Provider finish / stop reason",
    )
    prompt_tokens: int = Field(ge=0, description="Input token count (denormalized)")
    completion_tokens: int = Field(
        ge=0,
        description="Output token count (denormalized)",
    )
    total_tokens: int = Field(ge=0, description="Total token count (denormalized)")

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        """Reject blank answers (empty string after strip)."""
        if not value.strip():
            raise ValueError("GeneratedAnswer.answer must be non-empty")
        return value

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        """Reject blank model ids."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("GeneratedAnswer.model must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def denormalized_tokens_must_match_usage(self) -> GeneratedAnswer:
        """Keep top-level token fields aligned with ``usage``."""
        if (
            self.prompt_tokens != self.usage.prompt_tokens
            or self.completion_tokens != self.usage.completion_tokens
            or self.total_tokens != self.usage.total_tokens
        ):
            raise ValueError(
                "Top-level token fields must match usage "
                f"(got prompt={self.prompt_tokens}, "
                f"completion={self.completion_tokens}, "
                f"total={self.total_tokens}; "
                f"usage={self.usage})"
            )
        return self

    @classmethod
    def from_parts(
        cls,
        *,
        answer: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str | None = None,
    ) -> GeneratedAnswer:
        """Build a ``GeneratedAnswer`` with consistent usage fields.

        Args:
            answer: Generated text.
            model: Model id.
            prompt_tokens: Input tokens.
            completion_tokens: Output tokens.
            finish_reason: Optional provider stop reason.

        Returns:
            Validated ``GeneratedAnswer`` instance.
        """
        total = prompt_tokens + completion_tokens
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
        return cls(
            answer=answer,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
