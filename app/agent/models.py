"""Agent DTOs — planner output, run results, tool invocation requests.

These models sit *above* evaluation primitives (``ToolContract``,
``ToolExecution``, ``EvaluationContext``). Agents produce them; evaluation
consumes the nested evaluation types without knowing LangGraph.

Sprint 12 adds ``PlannerDecision`` — a typed planner contract with
confidence and execution strategy — while keeping ``PlannerOutput`` for
backward compatibility with Sprint 11 graph state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.context import EvaluationContext
from app.evaluation.tool_validation.report import ToolValidationReport
from app.evaluation.tool_validation.tool_contract import ToolContract
from app.evaluation.tool_validation.tool_execution import ToolExecution


class ToolInvocation(BaseModel):
    """One planned tool call with concrete arguments."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Registered tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for this invocation",
    )
    order: int = Field(default=0, ge=0, description="0-based execution order")

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank tool names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be non-empty")
        return cleaned


class ExecutionStrategy(str, Enum):
    """How the router should execute selected tools."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class PlannerDecision(BaseModel):
    """Typed planner decision (Sprint 12) — preferred over generic dicts.

    Fields:
        selected_tools: Tools chosen for execution (ordered).
        reasoning: Why this plan was chosen.
        confidence: Planner confidence in ``[0, 1]``.
        tool_contracts: Declarative contracts for evaluation.
        execution_strategy: sequential | parallel | conditional.
        required_tools: Must-run tools.
        optional_tools: Nice-to-have tools.
        alternative_tools: Substitutable tools.
        invocations: Concrete argument payloads.
        metadata: Future extensibility bag.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    selected_tools: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", description="Why this plan was chosen")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Planner confidence in [0, 1]",
    )
    tool_contracts: list[ToolContract] = Field(default_factory=list)
    execution_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.SEQUENTIAL,
    )
    required_tools: list[str] = Field(default_factory=list)
    optional_tools: list[str] = Field(default_factory=list)
    alternative_tools: list[str] = Field(default_factory=list)
    invocations: list[ToolInvocation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_planner_output(self) -> PlannerOutput:
        """Convert to Sprint 11 ``PlannerOutput`` for graph compatibility."""
        return PlannerOutput(
            required_tools=list(self.required_tools or self.selected_tools),
            optional_tools=list(self.optional_tools),
            alternative_tools=list(self.alternative_tools),
            execution_order=list(self.selected_tools),
            invocations=list(self.invocations),
            tool_contracts=list(self.tool_contracts),
            rationale=self.reasoning,
            planner_prompt=self.metadata.get("planner_prompt"),
            planner_response=self.metadata.get("planner_response"),
            direct_answer=self.metadata.get("direct_answer"),
            intent_route=self.metadata.get("intent_route"),
            pending_action=self.metadata.get("pending_action"),
        )

    @classmethod
    def from_planner_output(
        cls,
        output: PlannerOutput,
        *,
        confidence: float = 1.0,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ) -> PlannerDecision:
        """Lift Sprint 11 ``PlannerOutput`` into a typed decision."""
        return cls(
            selected_tools=list(output.selected_tools),
            reasoning=output.rationale,
            confidence=confidence,
            tool_contracts=list(output.tool_contracts),
            execution_strategy=execution_strategy,
            required_tools=list(output.required_tools),
            optional_tools=list(output.optional_tools),
            alternative_tools=list(output.alternative_tools),
            invocations=list(output.invocations),
            metadata={
                "planner_prompt": output.planner_prompt,
                "planner_response": output.planner_response,
                "direct_answer": output.direct_answer,
                "intent_route": output.intent_route,
                "pending_action": output.pending_action,
            },
        )


class PlannerOutput(BaseModel):
    """Structured planner decision for one agent turn.

    Attributes:
        required_tools: Tools that must run for a correct answer.
        optional_tools: Tools that may run but are not required.
        alternative_tools: Tools that may substitute for a required tool
            (same capability family — e.g. ``search`` vs ``search_docs``).
        execution_order: Ordered tool names the router should execute.
        invocations: Concrete per-call argument payloads.
        tool_contracts: Declarative contracts for evaluation.
        rationale: Short human-readable planning explanation.
        planner_prompt: Optional prompt sent to an LLM planner.
        planner_response: Optional raw LLM planner response.
    """

    model_config = ConfigDict(extra="forbid")

    required_tools: list[str] = Field(default_factory=list)
    optional_tools: list[str] = Field(default_factory=list)
    alternative_tools: list[str] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    invocations: list[ToolInvocation] = Field(default_factory=list)
    tool_contracts: list[ToolContract] = Field(default_factory=list)
    rationale: str = Field(default="", description="Why this plan was chosen")
    planner_prompt: str | None = Field(
        default=None,
        description="LLM planner prompt when used",
    )
    planner_response: str | None = Field(
        default=None,
        description="LLM planner response when used",
    )
    direct_answer: str | None = Field(
        default=None,
        description="User-facing answer when no tools should run",
    )
    intent_route: str | None = Field(
        default=None,
        description="Conversational / domain route selected before tools",
    )
    pending_action: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional human-in-the-loop draft (e.g. pending leave request) "
            "that must not execute until confirmed"
        ),
    )

    @property
    def selected_tools(self) -> list[str]:
        """Tools selected for execution (execution order, else required)."""
        if self.execution_order:
            return list(self.execution_order)
        return list(self.required_tools)

    def to_decision(
        self,
        *,
        confidence: float = 1.0,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    ) -> PlannerDecision:
        """Convert to Sprint 12 ``PlannerDecision``."""
        return PlannerDecision.from_planner_output(
            self,
            confidence=confidence,
            execution_strategy=execution_strategy,
        )


class AgentRequest(BaseModel):
    """Inbound request for one LangGraph agent turn via the HTTP API.

    Attributes:
        question: User question.
        metadata: Optional run metadata (e.g. scenario id).
        expected_answer: Optional golden answer for EvaluationContext.
        expected_sources: Optional expected sources.
        validate_tools: When True, run ``ToolValidator`` on contracts.
        correlation_id: Optional correlation id.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="User question for the agent")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional run metadata",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Optional golden / expected answer",
    )
    expected_sources: list[str] | None = Field(
        default=None,
        description="Optional expected source identifiers",
    )
    validate_tools: bool = Field(
        default=True,
        description="Whether to run tool validation",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Optional correlation id",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject blank questions."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-empty")
        return cleaned


class AgentRunResult(BaseModel):
    """Outbound result of one LangGraph agent run.

    Attributes:
        question: Original user question.
        final_answer: Agent answer text.
        tool_executions: Observed ``ToolExecution`` records.
        planner_output: Structured planner decision (Sprint 11).
        planner_decision: Typed planner decision (Sprint 12).
        evaluation_context: Filled evaluation aggregate.
        tool_validation: Optional tool-validation report.
        correlation_id: Request correlation id.
        trace_id: LangSmith / local trace id.
        run_id: LangSmith / local run id.
        run_url: Optional LangSmith UI URL.
        latency_ms: End-to-end agent latency.
        metadata: Free-form extras.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    final_answer: str = ""
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    planner_output: PlannerOutput | None = None
    planner_decision: PlannerDecision | None = None
    evaluation_context: EvaluationContext | None = None
    tool_validation: ToolValidationReport | None = None
    correlation_id: str = ""
    trace_id: str | None = None
    run_id: str | None = None
    run_url: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ToolInvocation",
    "ExecutionStrategy",
    "PlannerDecision",
    "PlannerOutput",
    "AgentRequest",
    "AgentRunResult",
]
