"""OpenAI Chat Completions LLMProvider Strategy.

Uses the official OpenAI Python SDK. Configuration comes from Settings
(``OPENAI_API_KEY``, ``OPENAI_MODEL``, ``TEMPERATURE``, ``MAX_TOKENS``).
The client is injectable so unit tests never hit the live API.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from app.config.settings import Settings, get_settings
from app.generation.exceptions import (
    GenerationError,
    GenerationTimeoutError,
    InvalidGenerationConfigError,
    InvalidModelError,
    MalformedResponseError,
    MissingAPIKeyError,
    NetworkError,
    RateLimitError,
)
from app.generation.llm_provider import LLMProvider
from app.generation.models import BuiltPrompt, GeneratedAnswer

logger = logging.getLogger(__name__)


class _ChatCompletionsLike(Protocol):
    """Minimal OpenAI client surface used by this Strategy."""

    @property
    def chat(self) -> Any: ...


ClientFactory = Callable[[str, float | None], _ChatCompletionsLike]


def _default_client_factory(
    api_key: str,
    timeout: float | None,
) -> _ChatCompletionsLike:
    """Construct a real OpenAI client (imported lazily for testability)."""
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"openai package is unavailable: {exc}") from exc
    return OpenAI(api_key=api_key, timeout=timeout)


class OpenAIProvider(LLMProvider):
    """Generate answers via OpenAI Chat Completions.

    Args:
        settings: Provides API key, model, temperature, max tokens, timeout.
        client: Optional pre-built OpenAI-compatible client (tests).
        client_factory: Optional factory ``(api_key, timeout) -> client``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: _ChatCompletionsLike | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory or _default_client_factory
        self._client = client
        self._validate_config()

    def generate(self, prompt: BuiltPrompt) -> GeneratedAnswer:
        """Call OpenAI Chat Completions and return a structured answer."""
        model = self._settings.openai_model
        temperature = self._settings.temperature
        max_tokens = self._settings.max_tokens

        prompt_length = len(prompt.system) + len(prompt.user)
        logger.info(
            "Generation started: model=%s temperature=%s max_tokens=%s "
            "prompt_chars=%s context_docs=%s context_chars=%s",
            model,
            temperature,
            max_tokens,
            prompt_length,
            prompt.context_document_count,
            prompt.context_char_length,
        )

        client = self._get_client()
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "Generation failed: model=%s latency_ms=%.1f error=%s",
                model,
                latency_ms,
                exc,
            )
            raise self._map_provider_error(exc) from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            answer = self._parse_response(response, fallback_model=model)
        except MalformedResponseError:
            logger.error(
                "Generation response malformed: model=%s latency_ms=%.1f",
                model,
                latency_ms,
            )
            raise

        logger.info(
            "Generation finished: model=%s finish_reason=%s "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s "
            "latency_ms=%.1f",
            answer.model,
            answer.finish_reason,
            answer.prompt_tokens,
            answer.completion_tokens,
            answer.total_tokens,
            latency_ms,
        )
        return answer

    def _get_client(self) -> _ChatCompletionsLike:
        """Return a cached client, constructing one on first use."""
        if self._client is not None:
            return self._client

        api_key = (self._settings.openai_api_key or "").strip()
        if not api_key:
            logger.error("Generation rejected: missing OpenAI API key")
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is missing or blank; cannot call OpenAI"
            )

        try:
            self._client = self._client_factory(
                api_key,
                self._settings.openai_timeout_seconds,
            )
        except GenerationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                f"Failed to construct OpenAI client: {exc}"
            ) from exc
        return self._client

    def _validate_config(self) -> None:
        """Fail fast on clearly invalid generation settings."""
        model = (self._settings.openai_model or "").strip()
        if not model:
            raise InvalidGenerationConfigError(
                "OPENAI_MODEL must be a non-empty string"
            )
        if not (0.0 <= self._settings.temperature <= 2.0):
            raise InvalidGenerationConfigError(
                f"TEMPERATURE must be between 0 and 2, got {self._settings.temperature}"
            )
        if self._settings.max_tokens <= 0:
            raise InvalidGenerationConfigError(
                f"MAX_TOKENS must be > 0, got {self._settings.max_tokens}"
            )

    def _parse_response(
        self,
        response: Any,
        *,
        fallback_model: str,
    ) -> GeneratedAnswer:
        """Map an OpenAI response object into ``GeneratedAnswer``."""
        try:
            choices = getattr(response, "choices", None)
            if not choices:
                raise MalformedResponseError(
                    "OpenAI response contained no choices"
                )
            choice = choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None) if message else None
            if content is None or not str(content).strip():
                raise MalformedResponseError(
                    "OpenAI response contained empty message content"
                )

            usage_obj = getattr(response, "usage", None)
            if usage_obj is None:
                raise MalformedResponseError(
                    "OpenAI response missing usage information"
                )

            prompt_tokens = int(getattr(usage_obj, "prompt_tokens", -1))
            completion_tokens = int(getattr(usage_obj, "completion_tokens", -1))
            if prompt_tokens < 0 or completion_tokens < 0:
                raise MalformedResponseError(
                    "OpenAI usage fields are missing or invalid"
                )

            model = str(getattr(response, "model", None) or fallback_model)
            finish_reason = getattr(choice, "finish_reason", None)
            finish_reason_str = str(finish_reason) if finish_reason else None

            return GeneratedAnswer.from_parts(
                answer=str(content),
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason_str,
            )
        except MalformedResponseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MalformedResponseError(
                f"Failed to parse OpenAI response: {exc}"
            ) from exc

    @staticmethod
    def _map_provider_error(exc: Exception) -> GenerationError:
        """Translate OpenAI SDK exceptions into domain errors."""
        # Import lazily so unit tests without openai still load this module.
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                AuthenticationError,
                NotFoundError,
                OpenAIError,
                PermissionDeniedError,
                RateLimitError as OpenAIRateLimitError,
            )
        except Exception:  # noqa: BLE001
            return GenerationError(f"LLM provider call failed: {exc}")

        if isinstance(exc, AuthenticationError):
            return MissingAPIKeyError(
                f"OpenAI authentication failed (check OPENAI_API_KEY): {exc}"
            )
        if isinstance(exc, (NotFoundError, PermissionDeniedError)):
            return InvalidModelError(
                f"OpenAI rejected the configured model or request: {exc}"
            )
        if isinstance(exc, OpenAIRateLimitError):
            return RateLimitError(f"OpenAI rate limit exceeded: {exc}")
        if isinstance(exc, APITimeoutError):
            return GenerationTimeoutError(f"OpenAI request timed out: {exc}")
        if isinstance(exc, APIConnectionError):
            return NetworkError(f"OpenAI network/connection failure: {exc}")
        if isinstance(exc, OpenAIError):
            return GenerationError(f"OpenAI API error: {exc}")
        return GenerationError(f"LLM provider call failed: {exc}")
