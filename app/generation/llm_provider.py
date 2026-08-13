"""Abstract LLM provider contract.

Converts a ``BuiltPrompt`` into a ``GeneratedAnswer`` without depending on
retrievers, vector stores, FastAPI, or evaluation frameworks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.generation.models import BuiltPrompt, GeneratedAnswer


class LLMProvider(ABC):
    """Strategy interface for LLM text generation.

    Implementations wrap a concrete vendor SDK (OpenAI, Gemini, Claude,
    Azure OpenAI, Ollama, …). Business logic depends only on this ABC so
    providers can be swapped without changing PromptBuilder or callers.
    """

    @abstractmethod
    def generate(self, prompt: BuiltPrompt) -> GeneratedAnswer:
        """Generate an answer from a formatted prompt.

        Args:
            prompt: System + user prompt produced by ``PromptBuilder``.

        Returns:
            Structured answer including model id, finish reason, and usage.

        Raises:
            MissingAPIKeyError: API credentials are missing.
            InvalidModelError: Configured model is unknown / rejected.
            RateLimitError: Provider rate limit or quota exceeded.
            GenerationTimeoutError: Request timed out.
            NetworkError: Connectivity / transport failure.
            MalformedResponseError: Response could not be parsed.
            GenerationError: Other provider or configuration failures.
        """
