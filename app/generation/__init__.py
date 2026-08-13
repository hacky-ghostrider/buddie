"""Generation stage — PromptBuilder + LLMProvider → GeneratedAnswer.



Sprint 7 foundation; Sprint 8 adds file-based prompt templates and token

estimation used by the RAG orchestrator. No streaming, prompt-injection

defense, or evaluation here.

"""



from app.generation.base import LLMProvider

from app.generation.exceptions import (

    EmptyQuestionError,

    GenerationError,

    GenerationTimeoutError,

    InvalidGenerationConfigError,

    InvalidModelError,

    MalformedResponseError,

    MissingAPIKeyError,

    NetworkError,

    PromptTemplateError,

    RateLimitError,

)

from app.generation.models import BuiltPrompt, GeneratedAnswer, TokenUsage

from app.generation.openai_provider import OpenAIProvider

from app.generation.prompt_builder import PromptBuilder



__all__ = [

    "LLMProvider",

    "PromptBuilder",

    "OpenAIProvider",

    "BuiltPrompt",

    "GeneratedAnswer",

    "TokenUsage",

    "GenerationError",

    "EmptyQuestionError",

    "MissingAPIKeyError",

    "InvalidModelError",

    "RateLimitError",

    "GenerationTimeoutError",

    "NetworkError",

    "MalformedResponseError",

    "InvalidGenerationConfigError",

    "PromptTemplateError",

]


