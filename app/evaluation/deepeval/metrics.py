"""DeepEval metric identifiers and metadata.

Centralizes the set of supported DeepEval metrics so adapters, registries,
and reports share one vocabulary — analogous to an enum of quality gates
in a Java QA platform.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DeepEvalMetricName(StrEnum):
    """Stable registry keys for DeepEval-backed metrics.

    These names appear in ``MetricResult.name`` and report aggregations.
    """

    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"


# Human-readable descriptions for reports and logs.
DEEPEVAL_METRIC_DESCRIPTIONS: Final[dict[DeepEvalMetricName, str]] = {
    DeepEvalMetricName.FAITHFULNESS: (
        "Whether the answer is faithful to the retrieved context "
        "(no unsupported claims)."
    ),
    DeepEvalMetricName.HALLUCINATION: (
        "Inverse hallucination quality: 1.0 means no hallucination "
        "(DeepEval raw hallucination score is inverted so higher is better)."
    ),
    DeepEvalMetricName.ANSWER_RELEVANCY: (
        "Whether the answer is relevant to the user question."
    ),
    DeepEvalMetricName.CONTEXTUAL_PRECISION: (
        "Whether relevant retrieved chunks are ranked ahead of irrelevant ones."
    ),
    DeepEvalMetricName.CONTEXTUAL_RECALL: (
        "Whether retrieved context covers the information needed for the "
        "expected answer."
    ),
}

# Metrics that require an expected / golden answer.
METRICS_REQUIRING_EXPECTED_ANSWER: Final[frozenset[DeepEvalMetricName]] = frozenset(
    {
        DeepEvalMetricName.CONTEXTUAL_PRECISION,
        DeepEvalMetricName.CONTEXTUAL_RECALL,
    }
)

# Metrics where DeepEval's native score is "higher = worse".
METRICS_INVERT_SCORE: Final[frozenset[DeepEvalMetricName]] = frozenset(
    {
        DeepEvalMetricName.HALLUCINATION,
    }
)

DEFAULT_DEEPEVAL_METRICS: Final[tuple[DeepEvalMetricName, ...]] = (
    DeepEvalMetricName.FAITHFULNESS,
    DeepEvalMetricName.HALLUCINATION,
    DeepEvalMetricName.ANSWER_RELEVANCY,
    DeepEvalMetricName.CONTEXTUAL_PRECISION,
    DeepEvalMetricName.CONTEXTUAL_RECALL,
)
