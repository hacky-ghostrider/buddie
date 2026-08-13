"""Agent tool abstractions — Protocol + registry (Strategy + DI)."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from app.agent.exceptions import AgentToolNotFoundError
from app.evaluation.tool_validation.tool_execution import ToolExecution

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentTool(Protocol):
    """Strategy interface for one agent-callable tool.

    Implementations return ``ToolExecution`` — never bypass evaluation.
    """

    @property
    def name(self) -> str:
        """Stable tool name used by planner / contracts / validator."""

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Run the tool and return a normalized ``ToolExecution``."""


class ToolRegistry:
    """Name → ``AgentTool`` catalog (plugin registry).

    Args:
        tools: Optional initial tools to register.
    """

    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools: dict[str, AgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        """Register or replace a tool by name."""
        self._tools[tool.name] = tool
        logger.debug("Registered agent tool: %s", tool.name)

    def get(self, name: str) -> AgentTool:
        """Resolve a tool or raise ``AgentToolNotFoundError``."""
        tool = self._tools.get(name)
        if tool is None:
            raise AgentToolNotFoundError(f"Tool not registered: {name!r}")
        return tool

    def has(self, name: str) -> bool:
        """Return whether ``name`` is registered."""
        return name in self._tools

    def names(self) -> list[str]:
        """Return registered tool names in insertion order."""
        return list(self._tools.keys())

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Lookup + execute a tool, returning ``ToolExecution``."""
        tool = self.get(name)
        return tool.execute(
            arguments,
            order=order,
            correlation_id=correlation_id,
            context=context,
        )


__all__ = ["AgentTool", "ToolRegistry"]
