"""AgentState — explicit LangGraph shared state.

WHY explicit state improves debugging
-------------------------------------
Chat-only agents bury decisions inside free-form messages. When a tool
fails or the wrong tool is chosen, engineers grep logs hoping to
reconstruct order, arguments, and latency.

``AgentState`` is the flight recorder: every node reads/writes typed
fields (planner output, contracts, executions, evaluation context,
correlation / trace ids). Reproducing a failure means inspecting one
state snapshot — the same idea as a typed ``TestContext`` in Java
automation rather than scraping Selenium console logs.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


def _replace_list(_existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer that replaces the list (last writer wins)."""
    return list(new)


def _append_list(existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer that appends items (for incremental history)."""
    return list(existing or []) + list(new or [])


class AgentState(TypedDict):
    """LangGraph state schema for the production single-agent graph.

    Required keys are always present after ``AgentService`` seeds the run.
    Optional keys are filled by planner / router / finalize nodes.
    """

    question: str
    messages: Annotated[list[dict[str, Any]], operator.add]
    correlation_id: str

    planner_output: NotRequired[dict[str, Any] | None]
    selected_tools: NotRequired[Annotated[list[str], _replace_list]]
    tool_contracts: NotRequired[Annotated[list[dict[str, Any]], _replace_list]]
    tool_results: NotRequired[Annotated[list[Any], _replace_list]]
    tool_execution_history: NotRequired[
        Annotated[list[dict[str, Any]], _append_list]
    ]
    evaluation_context: NotRequired[dict[str, Any] | None]
    final_answer: NotRequired[str]
    trace_id: NotRequired[str | None]
    run_id: NotRequired[str | None]
    run_url: NotRequired[str | None]
    error: NotRequired[str | None]
    metadata: NotRequired[dict[str, Any]]


def empty_agent_state(
    *,
    question: str,
    correlation_id: str,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    """Build the initial state seed for a graph invocation.

    Args:
        question: User question.
        correlation_id: Correlation id for logs / traces.
        metadata: Optional run metadata.

    Returns:
        Initial ``AgentState``.
    """
    return AgentState(
        question=question,
        messages=[{"role": "user", "content": question}],
        correlation_id=correlation_id,
        planner_output=None,
        selected_tools=[],
        tool_contracts=[],
        tool_results=[],
        tool_execution_history=[],
        evaluation_context=None,
        final_answer="",
        trace_id=None,
        run_id=None,
        run_url=None,
        error=None,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "AgentState",
    "empty_agent_state",
]
