"""Domain exceptions for the generation stage."""


class GenerationError(Exception):
    """Base error for all generation failures."""


class EmptyQuestionError(GenerationError):
    """Raised when generation receives blank question text."""


class MissingAPIKeyError(GenerationError):
    """Raised when the LLM provider API key is missing or blank."""


class InvalidModelError(GenerationError):
    """Raised when the configured model id is rejected by the provider."""


class RateLimitError(GenerationError):
    """Raised when the provider returns a rate-limit / quota error."""


class GenerationTimeoutError(GenerationError):
    """Raised when the provider call exceeds the configured timeout."""


class NetworkError(GenerationError):
    """Raised when the provider cannot be reached (DNS, TLS, connection)."""


class MalformedResponseError(GenerationError):
    """Raised when the provider response cannot be parsed into GeneratedAnswer."""


class InvalidGenerationConfigError(GenerationError):
    """Raised when generation settings are invalid at runtime."""


class PromptTemplateError(GenerationError):
    """Raised when a prompt template cannot be loaded or is invalid."""
