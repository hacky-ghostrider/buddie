"""LangGraph agent package — planner, router, tools, evaluation integration.

Sprint 11 delivers a single production-quality agent. RAG is exposed as
tools (``search_docs`` / ``summarize``) that reuse ``RAGService``. Evaluation
continues to flow through ``ToolExecution`` → ``EvaluationContext`` →
``ToolValidator`` / DeepEval / LangSmith without architecture changes.
"""

from app.agent.exceptions import (
    AgentError,
    AgentPlanningError,
    AgentRoutingError,
    AgentStateError,
    AgentToolExecutionError,
    AgentToolNotFoundError,
)
from app.agent.graph import build_agent_graph, build_default_graph
from app.agent.models import (
    AgentRunResult,
    ExecutionStrategy,
    PlannerDecision,
    PlannerOutput,
    ToolInvocation,
)
from app.agent.planner import Planner, RuleBasedPlanner
from app.agent.router import ToolRouter
from app.agent.service import AgentService
from app.agent.state import AgentState, empty_agent_state
from app.agent.tools import (
    CalculatorTool,
    SearchDocsTool,
    SearchTool,
    SummarizeTool,
    ToolRegistry,
    build_default_tool_registry,
)

__all__ = [
    "AgentError",
    "AgentPlanningError",
    "AgentRoutingError",
    "AgentStateError",
    "AgentToolExecutionError",
    "AgentToolNotFoundError",
    "AgentRunResult",
    "AgentService",
    "AgentState",
    "ExecutionStrategy",
    "Planner",
    "PlannerDecision",
    "PlannerOutput",
    "RuleBasedPlanner",
    "ToolInvocation",
    "ToolRouter",
    "ToolRegistry",
    "CalculatorTool",
    "SearchDocsTool",
    "SearchTool",
    "SummarizeTool",
    "build_agent_graph",
    "build_default_graph",
    "build_default_tool_registry",
    "empty_agent_state",
]
