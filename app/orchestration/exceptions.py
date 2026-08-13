"""Domain exceptions for the RAG orchestration stage.

These wrap / classify failures from Retriever, PromptBuilder, and LLMProvider
so the API layer can map them to HTTP responses without knowing vendor details.
"""


class OrchestrationError(Exception):
    """Base error for all RAG orchestration failures."""


class EmptyRAGQuestionError(OrchestrationError):
    """Raised when the orchestrator receives a blank question."""


class RetrievalOrchestrationError(OrchestrationError):
    """Raised when retrieval fails inside the RAG pipeline."""


class PromptOrchestrationError(OrchestrationError):
    """Raised when prompt template loading or prompt building fails."""


class LLMOrchestrationError(OrchestrationError):
    """Raised when the LLM provider fails (timeout, network, API error, …)."""


class UnexpectedOrchestrationError(OrchestrationError):
    """Raised for unexpected failures inside the orchestrator."""
