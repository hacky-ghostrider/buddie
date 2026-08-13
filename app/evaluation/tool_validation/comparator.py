"""Comparators for tool names, arguments, counts, order, and latency.

Pure functions — easy to unit test, no I/O. Used by ``ToolValidator``.
"""

from __future__ import annotations

from typing import Any

from app.evaluation.tool_validation.models import ActualToolCall, ToolCallExpectation


def arguments_match(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    exact: bool = False,
) -> bool:
    """Compare argument maps.

    Args:
        expected: Expected arguments.
        actual: Actual arguments.
        exact: When True, require identical key sets and values.
            When False, every expected key/value must appear in actual
            (actual may have extras) — useful for agent frameworks that
            inject bookkeeping fields.

    Returns:
        ``True`` when arguments satisfy the comparison mode.
    """
    if exact:
        return expected == actual
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            return False
    return True


def filter_matching_calls(
    expectation: ToolCallExpectation,
    actual_calls: list[ActualToolCall],
) -> list[ActualToolCall]:
    """Return actual calls that match name (+ args) for an expectation.

    Order and count are validated separately by the validator.
    """
    matches: list[ActualToolCall] = []
    for call in actual_calls:
        if call.tool_name != expectation.tool_name:
            continue
        if not arguments_match(
            expectation.arguments,
            call.arguments,
            exact=expectation.require_exact_arguments,
        ):
            continue
        matches.append(call)
    return matches


def order_is_satisfied(
    expectation: ToolCallExpectation,
    matched_calls: list[ActualToolCall],
) -> bool:
    """Return whether at least one matched call sits at the expected index."""
    if expectation.order is None:
        return True
    return any(call.order == expectation.order for call in matched_calls)


def count_is_satisfied(
    expectation: ToolCallExpectation,
    matched_count: int,
) -> bool:
    """Return whether matched count is within ``[min_count, max_count]``."""
    if matched_count < expectation.min_count:
        return False
    if expectation.max_count is not None and matched_count > expectation.max_count:
        return False
    return True


def latency_is_satisfied(
    expectation: ToolCallExpectation,
    matched_calls: list[ActualToolCall],
) -> bool:
    """Return whether all matched calls respect ``max_latency_ms`` when set."""
    if expectation.max_latency_ms is None:
        return True
    for call in matched_calls:
        if call.latency_ms is None:
            return False
        if call.latency_ms > expectation.max_latency_ms:
            return False
    return True


__all__ = [
    "arguments_match",
    "filter_matching_calls",
    "order_is_satisfied",
    "count_is_satisfied",
    "latency_is_satisfied",
]
