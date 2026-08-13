"""End-to-end RAG orchestrator.

``RAGService`` is the application service that sequences Retriever →
PromptBuilder → LLMProvider. It depends only on abstractions — never on
OpenAI or Chroma concrete types — so unit tests can mock every collaborator.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.config.settings import Settings, get_settings
from app.generation.exceptions import (
    EmptyQuestionError,
    GenerationError,
    GenerationTimeoutError,
)
from app.generation.llm_provider import LLMProvider
from app.generation.models import BuiltPrompt, GeneratedAnswer
from app.generation.prompt_builder import PromptBuilder
from app.orchestration.exceptions import (
    EmptyRAGQuestionError,
    LLMOrchestrationError,
    PromptOrchestrationError,
    RetrievalOrchestrationError,
    UnexpectedOrchestrationError,
)
from app.orchestration.models import LatencyBreakdown, RAGRequest, RAGResponse
from app.retrieval.base import Retriever
from app.retrieval.exceptions import EmptyQueryError, RetrievalError
from app.retrieval.models import RetrievedDocument

logger = logging.getLogger(__name__)


class RAGService:
    """Orchestrate one grounded RAG query end-to-end.

    Responsibilities:
        1. Validate the request.
        2. Retrieve relevant documents.
        3. Build a grounded prompt from templates.
        4. Call the LLM provider.
        5. Assemble ``RAGResponse`` with latency and token metadata.

    Args:
        retriever: Abstraction used to fetch evidence chunks.
        prompt_builder: Abstraction used to format system + user prompts.
        llm_provider: Abstraction used to generate the answer.
        settings: Provides RAG defaults and context-token threshold.
    """

    def __init__(
        self,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._settings = settings or get_settings()

    def query(self, request: RAGRequest) -> RAGResponse:
        """Execute a full RAG pipeline for ``request``.

        Args:
            request: Validated inbound RAG request.

        Returns:
            Structured answer with evidence and observability metadata.

        Raises:
            EmptyRAGQuestionError: Question is blank.
            RetrievalOrchestrationError: Retriever failed.
            PromptOrchestrationError: Prompt building / template loading failed.
            LLMOrchestrationError: LLM provider failed (including timeouts).
            UnexpectedOrchestrationError: Any other unexpected failure.
        """
        correlation_id = str(uuid.uuid4())
        total_started = time.perf_counter()

        question = self._validate_question(request.question, correlation_id)
        top_k = (
            request.top_k
            if request.top_k is not None
            else self._settings.rag_default_top_k
        )
        score_threshold = (
            request.score_threshold
            if request.score_threshold is not None
            else self._settings.rag_default_score_threshold
        )

        logger.info(
            "RAG request received: correlation_id=%s question_preview=%r "
            "top_k=%s score_threshold=%s has_metadata_filters=%s",
            correlation_id,
            question[:80],
            top_k,
            score_threshold,
            request.metadata_filters is not None,
        )

        try:
            documents, retrieval_ms = self._retrieve(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filters=request.metadata_filters,
                correlation_id=correlation_id,
            )
            prompt, prompt_ms, estimated_tokens, context_exceeded = self._build_prompt(
                question=question,
                documents=documents,
                correlation_id=correlation_id,
            )
            answer, llm_ms = self._generate(
                prompt=prompt,
                correlation_id=correlation_id,
            )
        except (
            EmptyRAGQuestionError,
            RetrievalOrchestrationError,
            PromptOrchestrationError,
            LLMOrchestrationError,
        ):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unexpected orchestration failure: correlation_id=%s error=%s",
                correlation_id,
                exc,
            )
            raise UnexpectedOrchestrationError(
                f"Unexpected RAG orchestration failure: {exc}"
            ) from exc

        total_ms = (time.perf_counter() - total_started) * 1000.0
        latency = LatencyBreakdown(
            retrieval_ms=retrieval_ms,
            prompt_build_ms=prompt_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        )

        logger.info(
            "RAG request completed: correlation_id=%s retrieved=%s "
            "estimated_prompt_tokens=%s prompt_tokens=%s completion_tokens=%s "
            "retrieval_ms=%.1f prompt_ms=%.1f llm_ms=%.1f total_ms=%.1f",
            correlation_id,
            len(documents),
            estimated_tokens,
            answer.prompt_tokens,
            answer.completion_tokens,
            retrieval_ms,
            prompt_ms,
            llm_ms,
            total_ms,
        )

        return self._build_response(
            question=question,
            answer=answer,
            documents=documents,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filters=request.metadata_filters,
            estimated_tokens=estimated_tokens,
            context_exceeded=context_exceeded,
            latency=latency,
            correlation_id=correlation_id,
        )

    def _validate_question(self, question: str, correlation_id: str) -> str:
        """Reject blank questions before touching downstream layers."""
        cleaned = question.strip() if question else ""
        if not cleaned:
            logger.error(
                "RAG request rejected: empty question correlation_id=%s",
                correlation_id,
            )
            raise EmptyRAGQuestionError("Question must be a non-empty string")
        return cleaned

    def _retrieve(
        self,
        *,
        question: str,
        top_k: int,
        score_threshold: float,
        metadata_filters: dict[str, Any] | None,
        correlation_id: str,
    ) -> tuple[list[RetrievedDocument], float]:
        """Call the Retriever and return documents plus elapsed ms."""
        started = time.perf_counter()
        try:
            documents = self._retriever.retrieve(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filters=metadata_filters,
            )
        except EmptyQueryError as exc:
            # Should not happen after orchestrator validation, but map cleanly.
            raise EmptyRAGQuestionError(str(exc)) from exc
        except RetrievalError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "Retrieval failed: correlation_id=%s duration_ms=%.1f error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise RetrievalOrchestrationError(
                f"Retrieval failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "Retrieval failed unexpectedly: correlation_id=%s "
                "duration_ms=%.1f error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise RetrievalOrchestrationError(
                f"Retrieval failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Retrieval duration: correlation_id=%s duration_ms=%.1f "
            "retrieved_count=%s",
            correlation_id,
            elapsed_ms,
            len(documents),
        )
        return documents, elapsed_ms

    def _build_prompt(
        self,
        *,
        question: str,
        documents: list[RetrievedDocument],
        correlation_id: str,
    ) -> tuple[BuiltPrompt, float, int, bool]:
        """Build the prompt, estimate tokens, and warn on context overflow."""
        started = time.perf_counter()
        try:
            prompt = self._prompt_builder.build(question, documents)
        except EmptyQuestionError as exc:
            raise EmptyRAGQuestionError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "Prompt build failed: correlation_id=%s duration_ms=%.1f error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise PromptOrchestrationError(
                f"Prompt build failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        estimated_tokens = self._prompt_builder.estimate_token_count(prompt)
        max_context = self._settings.max_context_tokens
        context_exceeded = estimated_tokens > max_context

        if context_exceeded:
            logger.warning(
                "Prompt exceeds configured context threshold: "
                "correlation_id=%s estimated_tokens=%s max_context_tokens=%s "
                "(no truncation applied)",
                correlation_id,
                estimated_tokens,
                max_context,
            )

        logger.info(
            "Prompt duration: correlation_id=%s duration_ms=%.1f "
            "estimated_tokens=%s context_docs=%s",
            correlation_id,
            elapsed_ms,
            estimated_tokens,
            prompt.context_document_count,
        )
        return prompt, elapsed_ms, estimated_tokens, context_exceeded

    def _generate(
        self,
        *,
        prompt: BuiltPrompt,
        correlation_id: str,
    ) -> tuple[GeneratedAnswer, float]:
        """Call the LLM provider and return the answer plus elapsed ms."""
        started = time.perf_counter()
        try:
            answer = self._llm_provider.generate(prompt)
        except GenerationTimeoutError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "LLM timed out: correlation_id=%s duration_ms=%.1f error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise LLMOrchestrationError(f"LLM timed out: {exc}") from exc
        except GenerationError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "LLM generation failed: correlation_id=%s duration_ms=%.1f "
                "error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise LLMOrchestrationError(f"LLM generation failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.error(
                "LLM generation failed unexpectedly: correlation_id=%s "
                "duration_ms=%.1f error=%s",
                correlation_id,
                elapsed_ms,
                exc,
            )
            raise LLMOrchestrationError(
                f"LLM generation failed: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Generation duration: correlation_id=%s duration_ms=%.1f "
            "prompt_tokens=%s completion_tokens=%s total_tokens=%s",
            correlation_id,
            elapsed_ms,
            answer.prompt_tokens,
            answer.completion_tokens,
            answer.total_tokens,
        )
        return answer, elapsed_ms

    @staticmethod
    def _build_response(
        *,
        question: str,
        answer: GeneratedAnswer,
        documents: list[RetrievedDocument],
        top_k: int,
        score_threshold: float,
        metadata_filters: dict[str, Any] | None,
        estimated_tokens: int,
        context_exceeded: bool,
        latency: LatencyBreakdown,
        correlation_id: str,
    ) -> RAGResponse:
        """Assemble the outbound ``RAGResponse`` contract."""
        avg_score = (
            sum(doc.score for doc in documents) / len(documents) if documents else 0.0
        )
        retrieval_metadata: dict[str, Any] = {
            "retrieved_count": len(documents),
            "top_k": top_k,
            "score_threshold": score_threshold,
            "average_score": avg_score,
            "metadata_filters": metadata_filters,
        }
        generation_metadata: dict[str, Any] = {
            "model": answer.model,
            "finish_reason": answer.finish_reason,
            "estimated_prompt_tokens": estimated_tokens,
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "total_tokens": answer.total_tokens,
            "context_exceeded_threshold": context_exceeded,
            "retrieval_latency_ms": latency.retrieval_ms,
            "prompt_build_latency_ms": latency.prompt_build_ms,
            "llm_latency_ms": latency.llm_ms,
            "total_latency_ms": latency.total_ms,
        }
        return RAGResponse(
            question=question,
            answer=answer.answer,
            retrieved_documents=documents,
            retrieval_metadata=retrieval_metadata,
            generation_metadata=generation_metadata,
            latency=latency,
            correlation_id=correlation_id,
        )
