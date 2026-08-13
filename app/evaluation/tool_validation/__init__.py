"""Tool validation framework — evaluate agent tool usage without implementing agents.

This package validates *expected vs actual* tool calls so Sprint 11 agents
(LangGraph, OpenAI Agents, CrewAI, AutoGen, MCP) can plug in later.

Sprint 10.2 adds ``ToolContract``, ``ToolExecution``, and ``ToolTraceMapper``
so LangSmith (and future runtimes) stay behind an anti-corruption layer.
"""

from app.evaluation.tool_validation.models import (
    ActualToolCall,
    ToolCallExpectation,
    ToolMatchResult,
)
from app.evaluation.tool_validation.report import ToolValidationReport
from app.evaluation.tool_validation.tool_contract import (
    ToolContract,
    contracts_from_golden_fields,
)
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionMetrics,
    ToolExecutionStatus,
)
from app.evaluation.tool_validation.tool_result import (
    CalculatorResultData,
    SearchResultData,
    ToolResult,
)
from app.evaluation.tool_validation.trace_mapper import ToolTraceMapper
from app.evaluation.tool_validation.validator import ToolValidator

__all__ = [
    "ActualToolCall",
    "ToolCallExpectation",
    "ToolMatchResult",
    "ToolValidationReport",
    "ToolValidator",
    "ToolContract",
    "ToolExecution",
    "ToolExecutionStatus",
    "ToolExecutionMetrics",
    "ToolResult",
    "CalculatorResultData",
    "SearchResultData",
    "ToolTraceMapper",
    "contracts_from_golden_fields",
]
