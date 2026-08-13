"""Evaluation Engine extension points — roadmap contracts (not full metrics).

This module declares the metric groups required by the Buddie evaluation
roadmap so later sprints can plug in DeepEval / RAGAS / custom judges without
rewriting agent or tool contracts.

IMPORTANT: Concrete scoring implementations are intentionally deferred.
Do not treat registered roadmap names as live CI gates yet.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class MetricGroup(str, Enum):
    """Logical buckets for the future Evaluation Engine."""

    RAG_RETRIEVAL = "rag_retrieval"
    RAG_GENERATION = "rag_generation"
    LLM_RESPONSE_QUALITY = "llm_response_quality"
    AGENT_TOOL_CALLING = "agent_tool_calling"
    SAFETY = "safety"
    CONVERSATIONAL_ROBUSTNESS = "conversational_robustness"
    NL2SQL = "nl2sql"
    PERFORMANCE = "performance"
    DATASET_EXPERIMENT = "dataset_experiment"
    CI_GATES = "ci_gates"


class MetricSpec(BaseModel):
    """Declarative specification for a future metric implementation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    group: MetricGroup
    description: str
    deepeval_compatible: bool = False
    ragas_compatible: bool = False
    requires_reference_answer: bool = False
    requires_retrieved_context: bool = False
    requires_tool_trace: bool = False


@runtime_checkable
class AgentTraceEvaluator(Protocol):
    """Extension point for agent / tool-calling evaluation."""

    def evaluate_tool_trace(
        self,
        *,
        expected_tools: list[str],
        actual_tools: list[str],
        expected_arguments: list[dict[str, Any]] | None = None,
        actual_arguments: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compare expected vs actual tool sequences / arguments."""


@runtime_checkable
class SafetyEvaluator(Protocol):
    """Extension point for safety / responsible-AI checks."""

    def evaluate_safety(
        self,
        *,
        query: str,
        answer: str,
        tool_calls: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return safety scores / flags for one turn."""


@runtime_checkable
class EvaluationExperimentRunner(Protocol):
    """Extension point for batch / regression / golden evaluation runs."""

    def run_experiment(
        self,
        *,
        dataset_id: str,
        thresholds: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a named experiment and return aggregate scores."""


# ---------------------------------------------------------------------------
# Roadmap catalog — names only; implementations land in later sprints
# ---------------------------------------------------------------------------

EVALUATION_METRIC_ROADMAP: list[MetricSpec] = [
    # A. RAG / retrieval
    MetricSpec(
        name="precision_at_k",
        group=MetricGroup.RAG_RETRIEVAL,
        description="Precision@K for retrieved chunks",
        ragas_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="recall_at_k",
        group=MetricGroup.RAG_RETRIEVAL,
        description="Recall@K for retrieved chunks",
        ragas_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="hit_at_k",
        group=MetricGroup.RAG_RETRIEVAL,
        description="Hit@K for gold documents",
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="mrr",
        group=MetricGroup.RAG_RETRIEVAL,
        description="Mean Reciprocal Rank",
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="ndcg_at_k",
        group=MetricGroup.RAG_RETRIEVAL,
        description="NDCG@K",
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="context_precision",
        group=MetricGroup.RAG_RETRIEVAL,
        description="RAGAS / DeepEval context precision",
        deepeval_compatible=True,
        ragas_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="context_recall",
        group=MetricGroup.RAG_RETRIEVAL,
        description="RAGAS / DeepEval context recall",
        deepeval_compatible=True,
        ragas_compatible=True,
        requires_reference_answer=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="context_relevance",
        group=MetricGroup.RAG_RETRIEVAL,
        description="Retrieved context relevance to the query",
        deepeval_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="retrieval_latency",
        group=MetricGroup.PERFORMANCE,
        description="Retrieval latency in milliseconds",
    ),
    # A/B generation groundedness
    MetricSpec(
        name="faithfulness",
        group=MetricGroup.RAG_GENERATION,
        description="Answer faithfulness to retrieved context",
        deepeval_compatible=True,
        ragas_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="answer_relevancy",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="Answer relevancy to the user query",
        deepeval_compatible=True,
        ragas_compatible=True,
    ),
    MetricSpec(
        name="answer_correctness",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="Answer correctness vs reference",
        deepeval_compatible=True,
        ragas_compatible=True,
        requires_reference_answer=True,
    ),
    MetricSpec(
        name="groundedness",
        group=MetricGroup.RAG_GENERATION,
        description="Factual grounding / groundedness",
        deepeval_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="hallucination_rate",
        group=MetricGroup.RAG_GENERATION,
        description="Hallucination rate (lower is better)",
        deepeval_compatible=True,
        requires_retrieved_context=True,
    ),
    MetricSpec(
        name="completeness",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="Response completeness",
    ),
    MetricSpec(
        name="conciseness",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="Response conciseness",
    ),
    MetricSpec(
        name="semantic_similarity",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="Semantic similarity to reference answer",
        requires_reference_answer=True,
    ),
    MetricSpec(
        name="geval_judge",
        group=MetricGroup.LLM_RESPONSE_QUALITY,
        description="LLM-as-a-Judge / GEval-style evaluation",
        deepeval_compatible=True,
    ),
    # C. Agent / tool-calling
    MetricSpec(
        name="intent_classification_accuracy",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Intent classification accuracy",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="route_selection_accuracy",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Route selection accuracy",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="tool_selection_accuracy",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Tool selection accuracy",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="tool_call_success_rate",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Tool-call success rate",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="tool_argument_accuracy",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Tool argument accuracy",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="tool_ordering_correctness",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Tool ordering correctness",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="multi_tool_workflow_success_rate",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Multi-tool workflow success rate",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="human_confirmation_compliance",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Write actions require explicit confirmation",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="unauthorized_tool_call_rate",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Rate of unauthorized / spoofed tool calls",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="tool_hallucination_rate",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Fabricated tool-call rate",
        requires_tool_trace=True,
    ),
    # D. Safety
    MetricSpec(
        name="toxicity",
        group=MetricGroup.SAFETY,
        description="Toxicity detection",
        deepeval_compatible=True,
    ),
    MetricSpec(
        name="bias",
        group=MetricGroup.SAFETY,
        description="Bias detection",
        deepeval_compatible=True,
    ),
    MetricSpec(
        name="prompt_injection_detection",
        group=MetricGroup.SAFETY,
        description="Prompt injection detection",
    ),
    MetricSpec(
        name="pii_leakage",
        group=MetricGroup.SAFETY,
        description="PII / sensitive-data leakage",
    ),
    MetricSpec(
        name="unauthorized_data_access",
        group=MetricGroup.SAFETY,
        description="Unauthorized employee-data access attempts",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="write_action_confirmation_compliance",
        group=MetricGroup.SAFETY,
        description="Write-action confirmation compliance",
        requires_tool_trace=True,
    ),
    # E. Conversational robustness
    MetricSpec(
        name="correct_route_rate",
        group=MetricGroup.CONVERSATIONAL_ROBUSTNESS,
        description="Correct route rate for greetings / noise / OOD",
    ),
    MetricSpec(
        name="graceful_fallback_rate",
        group=MetricGroup.CONVERSATIONAL_ROBUSTNESS,
        description="Graceful fallback rate",
    ),
    MetricSpec(
        name="unwanted_rag_activation_rate",
        group=MetricGroup.CONVERSATIONAL_ROBUSTNESS,
        description="Unwanted RAG activation rate",
    ),
    MetricSpec(
        name="unwanted_tool_call_rate",
        group=MetricGroup.CONVERSATIONAL_ROBUSTNESS,
        description="Unwanted tool-call rate",
        requires_tool_trace=True,
    ),
    # F. NL2SQL (Learning Lab / Evaluation — not Buddie employee workflow)
    MetricSpec(
        name="sql_validity",
        group=MetricGroup.NL2SQL,
        description="SQL validity / syntax correctness",
    ),
    MetricSpec(
        name="execution_accuracy",
        group=MetricGroup.NL2SQL,
        description="NL2SQL execution accuracy",
    ),
    MetricSpec(
        name="unsafe_sql_detection",
        group=MetricGroup.NL2SQL,
        description="Unsafe SQL detection / read-only enforcement",
    ),
    # G. Performance
    MetricSpec(
        name="end_to_end_latency",
        group=MetricGroup.PERFORMANCE,
        description="End-to-end agent latency",
    ),
    MetricSpec(
        name="tool_latency",
        group=MetricGroup.PERFORMANCE,
        description="Per-tool latency",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="mcp_latency",
        group=MetricGroup.PERFORMANCE,
        description="MCP protocol / transport latency",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="mcp_tool_success_rate",
        group=MetricGroup.AGENT_TOOL_CALLING,
        description="Success rate for tools invoked through MCP",
        requires_tool_trace=True,
    ),
    MetricSpec(
        name="token_usage",
        group=MetricGroup.PERFORMANCE,
        description="Token usage / estimated cost",
    ),
]


class EvaluationRoadmapCatalog:
    """Read-only catalog of planned evaluation metrics / extension points."""

    def __init__(self, specs: list[MetricSpec] | None = None) -> None:
        self._specs = list(specs or EVALUATION_METRIC_ROADMAP)

    def list_groups(self) -> list[str]:
        """Return distinct metric group names."""
        return sorted({spec.group.value for spec in self._specs})

    def list_metrics(self, *, group: MetricGroup | str | None = None) -> list[MetricSpec]:
        """Return metric specs, optionally filtered by group."""
        if group is None:
            return list(self._specs)
        group_value = group.value if isinstance(group, MetricGroup) else str(group)
        return [spec for spec in self._specs if spec.group.value == group_value]

    def metric_names(self) -> list[str]:
        """Return all planned metric names."""
        return [spec.name for spec in self._specs]

    def as_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Group specs for documentation / future CI wiring."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for spec in self._specs:
            grouped.setdefault(spec.group.value, []).append(
                spec.model_dump(mode="json")
            )
        return grouped


# Future CI gate placeholders (not enforced in this sprint).
CI_EVALUATION_GATE_PLACEHOLDERS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "hit_at_k": 0.70,
    "tool_selection_accuracy": 0.80,
    "multi_tool_workflow_success_rate": 0.80,
    "safety_tests_pass_rate": 1.0,
}


__all__ = [
    "MetricGroup",
    "MetricSpec",
    "AgentTraceEvaluator",
    "SafetyEvaluator",
    "EvaluationExperimentRunner",
    "EVALUATION_METRIC_ROADMAP",
    "EvaluationRoadmapCatalog",
    "CI_EVALUATION_GATE_PLACEHOLDERS",
]
