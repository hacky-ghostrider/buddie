"""Quality recommendations — actionable remediations for failed gates.

Every failed or warned rule should tell an engineer *what to do next*,
not just that a number was below threshold. Recommendations are mapped by
``QualityRuleId`` so reports stay consistent across runs.
"""

from __future__ import annotations

from app.evaluation.quality.decision import QualityRecommendation
from app.evaluation.quality.models import QualityRuleId, QualityRuleResult, RuleSeverity


_RECOMMENDATION_CATALOG: dict[QualityRuleId, QualityRecommendation] = {
    QualityRuleId.MIN_FAITHFULNESS: QualityRecommendation(
        rule_id=QualityRuleId.MIN_FAITHFULNESS.value,
        category="retrieval",
        message="Low Faithfulness — answer is not sufficiently grounded in context",
        actions=[
            "Increase retrieval Top-K",
            "Tighten score threshold to prefer higher-quality chunks",
            "Review chunking size/overlap for context completeness",
            "Add explicit citation / grounding instructions to the prompt",
        ],
        priority="high",
    ),
    QualityRuleId.MAX_HALLUCINATION: QualityRecommendation(
        rule_id=QualityRuleId.MAX_HALLUCINATION.value,
        category="prompt",
        message="Hallucination above ceiling — model invents unsupported claims",
        actions=[
            "Review prompt template (refuse when context is insufficient)",
            "Lower temperature / sampling randomness",
            "Require answers to quote or paraphrase retrieved passages only",
            "Expand golden expected sources for the failing question",
        ],
        priority="high",
    ),
    QualityRuleId.MIN_ANSWER_RELEVANCY: QualityRecommendation(
        rule_id=QualityRuleId.MIN_ANSWER_RELEVANCY.value,
        category="generation",
        message="Low Answer Relevancy — response does not address the question",
        actions=[
            "Clarify the user-facing instruction in the prompt template",
            "Verify the planner selected the correct tools for the intent",
            "Check whether retrieved context matches the question topic",
        ],
        priority="high",
    ),
    QualityRuleId.MIN_CONTEXT_PRECISION: QualityRecommendation(
        rule_id=QualityRuleId.MIN_CONTEXT_PRECISION.value,
        category="retrieval",
        message="Low Context Precision — retrieved chunks include noise",
        actions=[
            "Raise similarity score threshold",
            "Reduce Top-K if noisy neighbors are diluting precision",
            "Improve embedding model or query rewriting",
        ],
        priority="medium",
    ),
    QualityRuleId.MIN_CONTEXT_RECALL: QualityRecommendation(
        rule_id=QualityRuleId.MIN_CONTEXT_RECALL.value,
        category="retrieval",
        message="Low Context Recall — key evidence was not retrieved",
        actions=[
            "Increase retrieval Top-K",
            "Revisit chunk boundaries so gold facts are not split awkwardly",
            "Confirm the source document was ingested and embedded",
        ],
        priority="medium",
    ),
    QualityRuleId.MAX_TOOL_FAILURES: QualityRecommendation(
        rule_id=QualityRuleId.MAX_TOOL_FAILURES.value,
        category="tools",
        message="Tool failures exceeded budget",
        actions=[
            "Inspect ToolExecution error / failure_reason fields",
            "Validate ToolContracts against actual planner invocations",
            "Add retries or harden tool argument validation",
        ],
        priority="high",
    ),
    QualityRuleId.MAX_TOOL_LATENCY: QualityRecommendation(
        rule_id=QualityRuleId.MAX_TOOL_LATENCY.value,
        category="latency",
        message="High tool latency — optimize retrieval or tool execution",
        actions=[
            "Profile ToolExecutionMetrics (queue vs execution time)",
            "Cache frequent retrieval queries where safe",
            "Reduce Top-K or parallelize independent tools",
        ],
        priority="medium",
    ),
    QualityRuleId.MAX_TOTAL_LATENCY: QualityRecommendation(
        rule_id=QualityRuleId.MAX_TOTAL_LATENCY.value,
        category="latency",
        message="High overall latency — end-to-end path exceeds budget",
        actions=[
            "Optimize retrieval or tool execution",
            "Review EvaluationTimeline for the slowest stage",
            "Consider smaller embedding / generation models for canaries",
        ],
        priority="medium",
    ),
    QualityRuleId.MAX_COST: QualityRecommendation(
        rule_id=QualityRuleId.MAX_COST.value,
        category="cost",
        message="Cost above budget — token usage is too high",
        actions=[
            "Reduce prompt / context size",
            "Lower max_tokens for generation",
            "Use a cheaper model for non-critical evaluation paths",
        ],
        priority="medium",
    ),
    QualityRuleId.MIN_OVERALL_SCORE: QualityRecommendation(
        rule_id=QualityRuleId.MIN_OVERALL_SCORE.value,
        category="score",
        message="Overall score below pass threshold",
        actions=[
            "Inspect per-metric DeepEval results for the weakest metric",
            "Compare against BenchmarkHistory for regressions",
            "Run RegressionEngine against the previous golden run",
        ],
        priority="high",
    ),
}


def recommendations_for_results(
    results: list[QualityRuleResult],
) -> list[QualityRecommendation]:
    """Build actionable recommendations for failed / warned rules.

    Args:
        results: Per-rule gate outcomes.

    Returns:
        Deduplicated list of ``QualityRecommendation`` ordered by priority.
    """
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    collected: list[QualityRecommendation] = []
    seen: set[str] = set()
    for result in results:
        if result.skipped or result.passed:
            continue
        # Warnings and fails both get recommendations.
        if result.severity not in {RuleSeverity.FAIL, RuleSeverity.WARNING}:
            continue
        catalog = _RECOMMENDATION_CATALOG.get(result.rule_id)
        if catalog is None:
            continue
        if catalog.rule_id in seen:
            continue
        seen.add(catalog.rule_id)
        collected.append(catalog)
    collected.sort(key=lambda item: priority_rank.get(item.priority, 9))
    return collected


__all__ = ["recommendations_for_results"]
