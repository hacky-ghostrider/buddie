"""Expectation helpers for tool validation.

Normalizes golden-dataset tool fields into ``ToolCallExpectation`` lists
so dataset schema and validator stay decoupled.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.tool_validation.models import ToolCallExpectation


def expectations_from_golden_fields(
    *,
    expected_tools: list[str] | None = None,
    expected_tool_arguments: list[dict[str, Any]] | None = None,
    expected_tool_order: list[str] | None = None,
    require_exact_arguments: bool = False,
) -> list[ToolCallExpectation]:
    """Build expectations from golden dataset columns.

    Rules:
        - If ``expected_tool_order`` is provided, it defines both names and order.
        - Else ``expected_tools`` defines names (order unconstrained unless
          indices are implied by argument list alignment).
        - ``expected_tool_arguments[i]`` aligns with tool ``i`` when present.

    Args:
        expected_tools: Expected tool names (unordered unless order also set).
        expected_tool_arguments: Per-tool argument dicts aligned by index.
        expected_tool_order: Ordered expected tool names.
        require_exact_arguments: Exact vs subset argument matching.

    Returns:
        List of ``ToolCallExpectation``.
    """
    args_list = list(expected_tool_arguments or [])
    if expected_tool_order:
        expectations: list[ToolCallExpectation] = []
        for index, name in enumerate(expected_tool_order):
            arguments = args_list[index] if index < len(args_list) else {}
            expectations.append(
                ToolCallExpectation(
                    tool_name=name,
                    arguments=dict(arguments),
                    order=index,
                    min_count=1,
                    max_count=None,
                    require_exact_arguments=require_exact_arguments,
                )
            )
        return expectations

    tools = list(expected_tools or [])
    expectations = []
    for index, name in enumerate(tools):
        arguments = args_list[index] if index < len(args_list) else {}
        expectations.append(
            ToolCallExpectation(
                tool_name=name,
                arguments=dict(arguments),
                order=None,
                min_count=1,
                max_count=None,
                require_exact_arguments=require_exact_arguments,
            )
        )
    return expectations


__all__ = ["expectations_from_golden_fields"]
