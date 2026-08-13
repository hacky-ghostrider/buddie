"""RAG orchestration — wires Retriever → PromptBuilder → LLMProvider.

Sprint 8 only. No evaluation metrics, DeepEval, RAGAS, streaming,
prompt-injection defense, hybrid search, or re-ranking.
"""

from app.orchestration.exceptions import (
    EmptyRAGQuestionError,
    LLMOrchestrationError,
    OrchestrationError,
    PromptOrchestrationError,
    RetrievalOrchestrationError,
    UnexpectedOrchestrationError,
)
from app.orchestration.models import LatencyBreakdown, RAGRequest, RAGResponse
from app.orchestration.rag_service import RAGService

__all__ = [
    "RAGService",
    "RAGRequest",
    "RAGResponse",
    "LatencyBreakdown",
    "OrchestrationError",
    "EmptyRAGQuestionError",
    "RetrievalOrchestrationError",
    "PromptOrchestrationError",
    "LLMOrchestrationError",
    "UnexpectedOrchestrationError",
]
