"""Offline extractive LLM — answers from retrieved context without OpenAI.

Used when ``OPENAI_API_KEY`` is blank so local Streamlit demos still return
a grounded response from the indexed corpus.
"""

from __future__ import annotations

import logging
import re

from app.generation.llm_provider import LLMProvider
from app.generation.models import BuiltPrompt, GeneratedAnswer

logger = logging.getLogger(__name__)

_CONTEXT_BLOCK_RE = re.compile(
    r"\[Document\s+(\d+)\]\s*id=(\S+)\s*[^\n]*\n(.*?)(?=\n\[Document\s+\d+\]|\n+Question:|\Z)",
    re.DOTALL,
)


class OfflineExtractiveProvider(LLMProvider):
    """Pick the most token-overlapping context chunk and quote it as the answer."""

    def __init__(self, model_name: str = "offline-extractive") -> None:
        self._model_name = model_name

    def generate(self, prompt: BuiltPrompt) -> GeneratedAnswer:
        question = prompt.question.strip()
        chunks = _extract_chunks(prompt.user)
        if not chunks:
            answer = (
                "I do not have enough information in the company knowledge "
                "to answer that."
            )
        else:
            best = max(chunks, key=lambda item: _overlap_score(question, item["text"]))
            # Keep technical chunk ids in logs only — never in employee chat.
            logger.info(
                "Offline extractive chose document_id=%s overlap_tokens=%s",
                best["id"],
                _overlap_score(question, best["text"]),
            )
            if _overlap_score(question, best["text"]) <= 0:
                answer = (
                    "I'm not able to help with that from the company knowledge "
                    "base. I can help with leave, holidays, benefits, and "
                    "company policies."
                )
            else:
                answer = (
                    "Here's what I found in the company policy:\n\n"
                    f"{best['text'].strip()}"
                )

        prompt_tokens = max(1, len(prompt.system.split()) + len(prompt.user.split()))
        completion_tokens = max(1, len(answer.split()))
        logger.info(
            "Offline extractive generation: model=%s chunks=%s",
            self._model_name,
            len(chunks),
        )
        return GeneratedAnswer.from_parts(
            answer=answer,
            model=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
        )


def build_llm_provider(settings) -> LLMProvider:
    """Prefer OpenAI when a key is present; otherwise use offline extractive."""
    from app.generation.openai_provider import OpenAIProvider

    if settings.openai_api_key.strip():
        return OpenAIProvider(settings=settings)
    logger.warning(
        "OPENAI_API_KEY blank; using OfflineExtractiveProvider for generation"
    )
    return OfflineExtractiveProvider()


def _extract_chunks(user_prompt: str) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for match in _CONTEXT_BLOCK_RE.finditer(user_prompt):
        text = match.group(3).strip()
        if text and "No retrieved documents" not in text:
            chunks.append({"id": match.group(2).strip(), "text": text})
    return chunks


def _overlap_score(question: str, text: str) -> int:
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not q_tokens:
        return 0
    return len(q_tokens & t_tokens)
