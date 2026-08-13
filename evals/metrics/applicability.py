"""Decide when LLM metrics are applicable for a Buddie evaluation case."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evals.metrics.config import (
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRICS_REQUIRING_RETRIEVAL_CONTEXT,
)

if TYPE_CHECKING:
    from evals.golden_dataset.models import BuddieGoldenCase
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

CONTEXTUAL_LLM_METRICS: frozenset[str] = frozenset(
    {
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    }
)


def has_meaningful_retrieval_context(case: DeepEvalCompatibleCase) -> bool:
    """True when runtime evidence contains non-blank retrieval/tool text."""
    return any(part.strip() for part in case.retrieval_context)


def annotated_expected_context(
    case: DeepEvalCompatibleCase,
    golden: BuddieGoldenCase | None = None,
) -> list[str]:
    """Golden reference context used for contextual metrics (never runtime evidence)."""
    if golden is not None:
        return list(golden.expected_context)
    return list(case.expected_context)


def has_annotated_expected_context(
    case: DeepEvalCompatibleCase,
    golden: BuddieGoldenCase | None = None,
) -> bool:
    """True when the golden case provides non-blank expected_context annotations."""
    return any(part.strip() for part in annotated_expected_context(case, golden))


def llm_metric_skip_reason(
    metric_name: str,
    case: DeepEvalCompatibleCase,
    golden: BuddieGoldenCase | None = None,
) -> str | None:
    """Return a skip reason when an LLM metric is not applicable, else ``None``."""
    if metric_name in METRICS_REQUIRING_RETRIEVAL_CONTEXT:
        if not has_meaningful_retrieval_context(case):
            return (
                "no runtime retrieval_context; golden expected_context is not used"
            )
    if metric_name in CONTEXTUAL_LLM_METRICS:
        if not has_annotated_expected_context(case, golden):
            return "no annotated expected_context for contextual metric"
    return None


__all__ = [
    "CONTEXTUAL_LLM_METRICS",
    "annotated_expected_context",
    "has_annotated_expected_context",
    "has_meaningful_retrieval_context",
    "llm_metric_skip_reason",
]
