"""Agent domain exceptions — typed failures for LangGraph orchestration."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for agent-layer failures."""


class AgentPlanningError(AgentError):
    """Planner could not produce a valid plan / tool contracts."""


class AgentRoutingError(AgentError):
    """Tool router failed to execute the planned tools."""


class AgentToolNotFoundError(AgentError):
    """Requested tool is not registered in the tool registry."""


class AgentToolExecutionError(AgentError):
    """A registered tool raised or returned a failed ``ToolExecution``."""


class AgentStateError(AgentError):
    """Agent state is missing required fields or is inconsistent."""


__all__ = [
    "AgentError",
    "AgentPlanningError",
    "AgentRoutingError",
    "AgentToolNotFoundError",
    "AgentToolExecutionError",
    "AgentStateError",
]
