"""Unit tests for offline extractive generation."""

from __future__ import annotations

from app.generation.models import BuiltPrompt
from app.generation.offline_provider import OfflineExtractiveProvider


def test_offline_extractive_stops_before_question() -> None:
    provider = OfflineExtractiveProvider()
    prompt = BuiltPrompt(
        system="Use context only.",
        user=(
            "Context:\n"
            "[Document 1] id=handbook-1 score=0.91 source=employee_handbook.md\n"
            "You are chatting with the RAG Evaluation Platform chatbot.\n\n"
            "Question:\n"
            "tell me who you are?\n\n"
            "Answer using only the context above."
        ),
        question="tell me who you are?",
        context_document_count=1,
        context_char_length=60,
    )
    result = provider.generate(prompt)
    assert "RAG Evaluation Platform chatbot" in result.answer
    assert "Answer using only the context above" not in result.answer
    assert "offline-extractive" in result.model
    assert "OPENAI_API_KEY" not in result.answer
    assert "Offline extractive" not in result.answer
