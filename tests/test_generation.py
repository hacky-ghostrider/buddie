"""Tests for the generation layer.

OpenAI is fully mocked — unit tests must never call the live API.
Mocking is required because live calls are non-deterministic, costly,
slow, and would leak secrets / burn quota in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.config.settings import Settings
from app.generation.exceptions import (
    EmptyQuestionError,
    GenerationTimeoutError,
    InvalidGenerationConfigError,
    InvalidModelError,
    MalformedResponseError,
    MissingAPIKeyError,
    NetworkError,
    RateLimitError,
)
from app.generation.models import BuiltPrompt, GeneratedAnswer, TokenUsage
from app.generation.openai_provider import OpenAIProvider
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.models import RetrievedDocument


def _doc(
    text: str,
    *,
    doc_id: str = "c1",
    score: float = 0.9,
    **meta: object,
) -> RetrievedDocument:
    return RetrievedDocument(
        id=doc_id,
        text=text,
        metadata={"file_name": "guide.pdf", **meta},
        score=score,
    )


def _openai_response(
    *,
    content: str = "Grounded answer from context.",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
    )


@pytest.fixture
def generation_settings() -> Settings:
    """Generation-focused settings with a dummy key for provider construction."""
    return Settings(
        app_env="test",
        openai_api_key="sk-test-not-real",
        openai_model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=256,
        openai_timeout_seconds=30.0,
    )


@pytest.fixture
def fake_client() -> MagicMock:
    """Mock OpenAI client that returns a deterministic completion."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response()
    return client


@pytest.fixture
def provider(
    generation_settings: Settings,
    fake_client: MagicMock,
) -> OpenAIProvider:
    """OpenAIProvider wired to the mocked client."""
    return OpenAIProvider(settings=generation_settings, client=fake_client)


@pytest.fixture
def sample_prompt() -> BuiltPrompt:
    """Minimal valid prompt for provider tests."""
    return BuiltPrompt(
        system="You are a grounded assistant.",
        user="Context:\n[Document 1]\nHello\n\nQuestion:\nWhat is this?",
        question="What is this?",
        context_document_count=1,
        context_char_length=20,
    )


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class TestPromptBuilder:
    def test_formats_system_and_user_prompt(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build(
            "What is chunking?",
            [_doc("Chunking splits long text.", doc_id="chunk-1", score=0.95)],
        )

        assert "ONLY the provided context" in prompt.system
        assert "Question:\nWhat is chunking?" in prompt.user
        assert "Context:" in prompt.user
        assert prompt.question == "What is chunking?"

    def test_injects_context_with_ids_and_scores(self) -> None:
        builder = PromptBuilder()
        docs = [
            _doc("First fact.", doc_id="a", score=0.91),
            _doc("Second fact.", doc_id="b", score=0.82),
        ]
        prompt = builder.build("Explain facts", docs)

        assert "[Document 1] id=a score=0.9100" in prompt.user
        assert "[Document 2] id=b score=0.8200" in prompt.user
        assert "First fact." in prompt.user
        assert "Second fact." in prompt.user
        assert prompt.context_document_count == 2
        assert prompt.context_char_length > 0

    def test_empty_context_injects_notice(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("Anything?", [])

        assert "No retrieved documents were provided" in prompt.user
        assert prompt.context_document_count == 0
        assert "Question:\nAnything?" in prompt.user

    def test_none_context_treated_as_empty(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("Anything?", None)
        assert prompt.context_document_count == 0
        assert "No retrieved documents were provided" in prompt.user

    def test_empty_question_raises(self) -> None:
        builder = PromptBuilder()
        with pytest.raises(EmptyQuestionError):
            builder.build("   ", [_doc("context")])

    def test_custom_system_prompt(self) -> None:
        builder = PromptBuilder(system_prompt="Answer briefly from context only.")
        prompt = builder.build("Q?", [_doc("A.")])
        assert prompt.system == "Answer briefly from context only."

    def test_blank_custom_system_prompt_rejected(self) -> None:
        with pytest.raises(ValueError, match="system_prompt"):
            PromptBuilder(system_prompt="  ")

    def test_estimate_token_count(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("What is chunking?", [_doc("Chunking splits text.")])
        estimate = builder.estimate_token_count(prompt)
        assert estimate > 0
        assert estimate == (
            (len(prompt.system) + len(prompt.user) + 3) // 4
        )

    def test_loads_templates_from_files(self) -> None:
        builder = PromptBuilder()
        prompt = builder.build("Q?", [_doc("A.")])
        assert "ONLY the provided context" in prompt.system
        assert "Context:" in prompt.user
        assert "Question:\nQ?" in prompt.user


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestGeneratedAnswerModel:
    def test_from_parts_aligns_usage(self) -> None:
        answer = GeneratedAnswer.from_parts(
            answer="Hello",
            model="gpt-4o-mini",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        assert answer.usage.total_tokens == 15
        assert answer.total_tokens == 15
        assert answer.prompt_tokens == 10
        assert answer.completion_tokens == 5

    def test_mismatched_usage_rejected(self) -> None:
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        with pytest.raises(ValidationError):
            GeneratedAnswer(
                answer="x",
                model="m",
                usage=usage,
                finish_reason="stop",
                prompt_tokens=9,
                completion_tokens=1,
                total_tokens=2,
            )

    def test_blank_answer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GeneratedAnswer.from_parts(
                answer="  ",
                model="gpt-4o-mini",
                prompt_tokens=1,
                completion_tokens=1,
            )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestGenerationSettings:
    def test_temperature_out_of_range(self) -> None:
        with pytest.raises(ValidationError, match="TEMPERATURE"):
            Settings(temperature=2.5)

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="MAX_TOKENS"):
            Settings(max_tokens=0)

    def test_openai_model_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="OPENAI_MODEL"):
            Settings(openai_model="  ")

    def test_valid_generation_defaults(self) -> None:
        settings = Settings()
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.temperature == 0.0
        assert settings.max_tokens == 1024


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_generate_parses_response(
        self,
        provider: OpenAIProvider,
        fake_client: MagicMock,
        sample_prompt: BuiltPrompt,
        generation_settings: Settings,
    ) -> None:
        result = provider.generate(sample_prompt)

        assert result.answer == "Grounded answer from context."
        assert result.model == "gpt-4o-mini"
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        assert result.total_tokens == 120
        assert result.usage.total_tokens == 120

        fake_client.chat.completions.create.assert_called_once()
        kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == generation_settings.openai_model
        assert kwargs["temperature"] == generation_settings.temperature
        assert kwargs["max_tokens"] == generation_settings.max_tokens
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][1]["role"] == "user"
        assert kwargs["messages"][0]["content"] == sample_prompt.system
        assert kwargs["messages"][1]["content"] == sample_prompt.user

    def test_usage_extraction(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response(
            prompt_tokens=42,
            completion_tokens=7,
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        result = provider.generate(sample_prompt)
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7
        assert result.total_tokens == 49

    def test_missing_api_key_on_call(self, sample_prompt: BuiltPrompt) -> None:
        settings = Settings(
            openai_api_key="",
            openai_model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=64,
        )
        provider = OpenAIProvider(settings=settings, client=None)
        with pytest.raises(MissingAPIKeyError):
            provider.generate(sample_prompt)

    def test_malformed_empty_choices(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            choices=[],
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(MalformedResponseError, match="no choices"):
            provider.generate(sample_prompt)

    def test_malformed_empty_content(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response(content="  ")
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(MalformedResponseError, match="empty message"):
            provider.generate(sample_prompt)

    def test_maps_rate_limit(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        from openai import RateLimitError as OpenAIRateLimitError

        client = MagicMock()
        client.chat.completions.create.side_effect = OpenAIRateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(RateLimitError):
            provider.generate(sample_prompt)

    def test_maps_timeout(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        from openai import APITimeoutError

        client = MagicMock()
        client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(GenerationTimeoutError):
            provider.generate(sample_prompt)

    def test_maps_network_error(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        from openai import APIConnectionError

        client = MagicMock()
        client.chat.completions.create.side_effect = APIConnectionError(
            request=MagicMock(),
            message="connection failed",
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(NetworkError):
            provider.generate(sample_prompt)

    def test_maps_invalid_model(
        self,
        generation_settings: Settings,
        sample_prompt: BuiltPrompt,
    ) -> None:
        from openai import NotFoundError

        client = MagicMock()
        client.chat.completions.create.side_effect = NotFoundError(
            message="model not found",
            response=MagicMock(status_code=404, headers={}),
            body=None,
        )
        provider = OpenAIProvider(settings=generation_settings, client=client)
        with pytest.raises(InvalidModelError):
            provider.generate(sample_prompt)

    def test_invalid_runtime_config_rejected(self) -> None:
        settings = MagicMock()
        settings.openai_model = "gpt-4o-mini"
        settings.temperature = 3.0
        settings.max_tokens = 100
        with pytest.raises(InvalidGenerationConfigError, match="TEMPERATURE"):
            OpenAIProvider(settings=settings, client=MagicMock())

    def test_end_to_end_builder_to_provider(
        self,
        provider: OpenAIProvider,
        fake_client: MagicMock,
    ) -> None:
        """PromptBuilder → OpenAIProvider without a live API."""
        builder = PromptBuilder()
        prompt = builder.build(
            "What is recursive chunking?",
            [_doc("Recursive chunking splits on separators.", doc_id="r1")],
        )
        answer = provider.generate(prompt)
        assert answer.answer
        called: dict[str, Any] = fake_client.chat.completions.create.call_args.kwargs
        assert "recursive chunking" in called["messages"][1]["content"].lower()
