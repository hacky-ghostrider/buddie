"""Deterministic semantic similarity — LLM-judge fallback for Buddie evals.

Uses token Jaccard overlap (no embedding model) so CI stays offline-friendly
when ``ENABLE_DEEPEVAL=false`` or Gemini is rate-limited.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from evals.metrics.config import DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD

if TYPE_CHECKING:
    from evals.golden_dataset.models import BuddieGoldenCase
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "from",
        "by",
        "at",
        "as",
        "it",
        "this",
        "that",
        "you",
        "your",
        "i",
        "my",
        "we",
        "our",
        "have",
        "has",
        "do",
        "does",
        "can",
        "will",
        "not",
        "answer",
        "buddie",
    }
)

_UNCERTAIN_MARKERS = ("[ANSWER_NOT_IN_CORPUS]", "[ANSWER_NOT_FULLY_SPECIFIED]")


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOP
    }


def _normalize_expected(text: str) -> str:
    cleaned = text.strip()
    for marker in _UNCERTAIN_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def token_jaccard_similarity(left: str, right: str) -> float:
    """Token Jaccard similarity in ``[0, 1]``."""
    a = _tokens(left)
    b = _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return round(inter / union, 6) if union else 0.0


def semantic_similarity_score(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
    *,
    threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
) -> tuple[float | None, float]:
    """Return ``(score, threshold)``; ``None`` score when not applicable."""
    expected = _normalize_expected(golden.expected_answer)
    actual = (case.actual_output or "").strip()
    if not expected or not actual or actual == "(empty)":
        return None, threshold

    if golden.expected_behavior in {
        "require_verification",
        "require_hitl_confirmation",
    }:
        # Structural checks matter more than token overlap for these flows.
        return None, threshold

    if golden.category == "adversarial_security":
        # Refusal wording varies; safety metrics own pass/fail.
        return None, threshold

    if any(m in golden.expected_answer for m in _UNCERTAIN_MARKERS):
        return None, threshold

    score = token_jaccard_similarity(actual, expected)
    return score, threshold


def semantic_similarity_passed(
    golden: BuddieGoldenCase,
    case: DeepEvalCompatibleCase,
    *,
    threshold: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
) -> float | None:
    """1.0 / 0.0 pass flag; ``None`` when N/A."""
    score, thresh = semantic_similarity_score(
        golden, case, threshold=threshold
    )
    if score is None:
        return None
    return 1.0 if score >= thresh else 0.0


__all__ = [
    "semantic_similarity_passed",
    "semantic_similarity_score",
    "token_jaccard_similarity",
]
