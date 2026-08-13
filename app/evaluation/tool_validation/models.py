"""Domain models for agent tool-call validation.

WHY custom models (not vendor schemas)
--------------------------------------
LangGraph, OpenAI Agents, CrewAI, AutoGen, and MCP each emit different
tool-call shapes. A neutral ``ActualToolCall`` / ``ToolCallExpectation``
lets one validator serve all frameworks after a thin mapper (Sprint 11).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolCallExpectation(BaseModel):
    """Expected tool invocation from a golden dataset / test case.

    Attributes:
        tool_name: Expected tool / function name.
        arguments: Expected argument map (subset match by default).
        order: Optional 0-based expected position in the call sequence.
        min_count: Minimum times this tool must appear.
        max_count: Optional maximum times this tool may appear.
        max_latency_ms: Optional per-call latency budget.
        require_exact_arguments: When True, args must equal exactly.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Expected tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected arguments (subset or exact)",
    )
    order: int | None = Field(
        default=None,
        ge=0,
        description="Optional expected index in the call sequence",
    )
    min_count: int = Field(default=1, ge=0, description="Minimum execution count")
    max_count: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum execution count",
    )
    max_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional max latency for matching calls",
    )
    require_exact_arguments: bool = Field(
        default=False,
        description="Require exact argument equality instead of subset",
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank tool names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be non-empty")
        return cleaned


class ActualToolCall(BaseModel):
    """One observed tool invocation from an agent runtime (future Sprint 11).

    Prefer producing ``ToolExecution`` via ``ToolTraceMapper`` and bridging
    with ``ToolTraceMapper.to_actual_tool_calls`` so validators stay
    vendor-blind. This model remains the Sprint 10 validator input.

    Attributes:
        tool_name: Actual tool that was invoked.
        arguments: Actual arguments passed.
        order: 0-based index in the observed sequence.
        latency_ms: Optional observed latency.
        success: Whether the tool returned successfully.
        metadata: Free-form extras (agent framework, node name, …).
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(description="Actual tool name")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Actual arguments",
    )
    order: int = Field(default=0, ge=0, description="Observed call index")
    latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Observed call latency in ms",
    )
    success: bool = Field(default=True, description="Tool call success flag")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional runtime metadata",
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank tool names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be non-empty")
        return cleaned


class ToolMatchResult(BaseModel):
    """Per-expectation comparison outcome.

    Attributes:
        expectation: The golden expectation that was checked.
        matched_calls: Actual calls that satisfied the expectation.
        passed: Whether this expectation passed.
        failures: Human-readable failure reasons.
    """

    model_config = ConfigDict(extra="forbid")

    expectation: ToolCallExpectation = Field(description="Expectation under test")
    matched_calls: list[ActualToolCall] = Field(
        default_factory=list,
        description="Actual calls matched to this expectation",
    )
    passed: bool = Field(description="Whether the expectation passed")
    failures: list[str] = Field(
        default_factory=list,
        description="Failure messages when not passed",
    )


__all__ = [
    "ToolCallExpectation",
    "ActualToolCall",
    "ToolMatchResult",
]
