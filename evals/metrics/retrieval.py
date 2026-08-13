"""Deterministic retrieval metrics for Buddie golden evaluation.

Ground truth: golden ``expected_context`` (annotated relevant evidence).
Predicted: runtime ``retrieval_context`` only — never substituted from goldens.

Metrics: Precision@K, Recall@K, Hit@K (K in {1,3,5}), MRR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Drop ultra-common glue tokens that appear in both tool JSON and gold notes.
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
        "true",
        "false",
        "null",
        "none",
        "days",
        "day",
        "status",
    }
)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOP
    }


def evidence_matches(retrieved: str, expected: str) -> bool:
    """Return True when a retrieved item covers an annotated expected item.

    Matching is deterministic and conservative:
    - normalized substring either direction, or
    - enough distinctive expected tokens appear in the retrieved text.
    """
    left = _normalize(retrieved)
    right = _normalize(expected)
    if not left or not right:
        return False
    if right in left or left in right:
        return True
    exp_tokens = _tokens(expected)
    if not exp_tokens:
        return False
    ret_tokens = _tokens(retrieved)
    if not ret_tokens:
        return False
    overlap = exp_tokens & ret_tokens
    # Require majority of expected tokens, with a small absolute floor.
    need = max(2, (len(exp_tokens) + 1) // 2)
    return len(overlap) >= min(need, len(exp_tokens))


def _relevant_flags(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
) -> list[bool]:
    expected = [e for e in expected_context if isinstance(e, str) and e.strip()]
    flags: list[bool] = []
    for item in retrieval_context:
        if not isinstance(item, str) or not item.strip():
            flags.append(False)
            continue
        flags.append(any(evidence_matches(item, exp) for exp in expected))
    return flags


def _expected_found_in_top_k(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
    k: int,
) -> list[bool]:
    top = list(retrieval_context)[: max(k, 0)]
    found: list[bool] = []
    for exp in expected_context:
        if not isinstance(exp, str) or not exp.strip():
            continue
        found.append(any(evidence_matches(item, exp) for item in top if item))
    return found


@dataclass(frozen=True)
class RetrievalMetricScores:
    """Nullable retrieval scores; None means not applicable."""

    precision_at_1: float | None = None
    precision_at_3: float | None = None
    precision_at_5: float | None = None
    recall_at_1: float | None = None
    recall_at_3: float | None = None
    recall_at_5: float | None = None
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_5: float | None = None
    mrr: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "precision_at_1": self.precision_at_1,
            "precision_at_3": self.precision_at_3,
            "precision_at_5": self.precision_at_5,
            "recall_at_1": self.recall_at_1,
            "recall_at_3": self.recall_at_3,
            "recall_at_5": self.recall_at_5,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
        }


def precision_at_k(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of top-K retrieved items that match any expected evidence."""
    if not expected_context:
        return None
    if k <= 0:
        return None
    top = list(retrieval_context)[:k]
    if not top:
        return 0.0
    flags = _relevant_flags(top, expected_context)
    return round(sum(1 for f in flags if f) / len(flags), 6)


def recall_at_k(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
    k: int,
) -> float | None:
    """Fraction of expected evidence items found in top-K retrieval."""
    expected = [e for e in expected_context if isinstance(e, str) and e.strip()]
    if not expected:
        return None
    if k <= 0:
        return None
    found = _expected_found_in_top_k(retrieval_context, expected, k)
    if not found:
        return 0.0
    return round(sum(1 for f in found if f) / len(found), 6)


def hit_at_k(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
    k: int,
) -> float | None:
    """1.0 if any expected evidence appears in top-K; else 0.0."""
    expected = [e for e in expected_context if isinstance(e, str) and e.strip()]
    if not expected:
        return None
    if k <= 0:
        return None
    found = _expected_found_in_top_k(retrieval_context, expected, k)
    return 1.0 if any(found) else 0.0


def mean_reciprocal_rank(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
) -> float | None:
    """Reciprocal rank of the first retrieved item matching any expected."""
    expected = [e for e in expected_context if isinstance(e, str) and e.strip()]
    if not expected:
        return None
    flags = _relevant_flags(retrieval_context, expected)
    for index, relevant in enumerate(flags, start=1):
        if relevant:
            return round(1.0 / index, 6)
    return 0.0


def compute_retrieval_metrics(
    retrieval_context: Sequence[str],
    expected_context: Sequence[str],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> RetrievalMetricScores:
    """Compute Precision/Recall/Hit@K and MRR for one case.

    When ``expected_context`` is empty, all scores are ``None`` (N/A).
    Empty ``retrieval_context`` with non-empty expected yields zeros.
    """
    expected = [e for e in expected_context if isinstance(e, str) and e.strip()]
    if not expected:
        return RetrievalMetricScores()

    ks = {int(k) for k in k_values}
    precision = {
        k: precision_at_k(retrieval_context, expected, k) for k in (1, 3, 5)
    }
    recall = {k: recall_at_k(retrieval_context, expected, k) for k in (1, 3, 5)}
    hit = {k: hit_at_k(retrieval_context, expected, k) for k in (1, 3, 5)}
    # Honor custom k_values for future use without changing the fixed schema.
    _ = ks
    return RetrievalMetricScores(
        precision_at_1=precision[1],
        precision_at_3=precision[3],
        precision_at_5=precision[5],
        recall_at_1=recall[1],
        recall_at_3=recall[3],
        recall_at_5=recall[5],
        hit_at_1=hit[1],
        hit_at_3=hit[3],
        hit_at_5=hit[5],
        mrr=mean_reciprocal_rank(retrieval_context, expected),
    )


__all__ = [
    "DEFAULT_K_VALUES",
    "RetrievalMetricScores",
    "compute_retrieval_metrics",
    "evidence_matches",
    "hit_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
