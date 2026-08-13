"""Calculator tool — safe arithmetic expression evaluation."""

from __future__ import annotations

import ast
import logging
import operator
import time
from datetime import datetime, timezone
from typing import Any

from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionMetrics,
    ToolExecutionStatus,
)
from app.evaluation.tool_validation.tool_result import (
    CalculatorResultData,
    ToolResult,
)

logger = logging.getLogger(__name__)

_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_arithmetic(expression: str) -> float | int:
    """Evaluate a safe arithmetic expression via AST (no ``eval``).

    Args:
        expression: Expression such as ``"(2 + 3) * 4"``.

    Returns:
        Numeric result.

    Raises:
        ValueError: Expression is empty or contains disallowed nodes.
        ZeroDivisionError: Division by zero.
    """
    cleaned = (expression or "").strip()
    if not cleaned:
        raise ValueError("expression must be non-empty")

    tree = ast.parse(cleaned, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool:
    """Agent tool for arithmetic expressions.

    Accepted arguments:
        * ``expression`` (preferred)
        * ``expr`` / ``query`` aliases
    """

    name = "calculator"

    def execute(
        self,
        arguments: dict[str, Any],
        *,
        order: int = 0,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ToolExecution:
        """Evaluate an arithmetic expression and return ``ToolExecution``."""
        del context  # unused
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()
        expression = str(
            arguments.get("expression")
            or arguments.get("expr")
            or arguments.get("query")
            or ""
        )
        try:
            result = evaluate_arithmetic(expression)
            latency_ms = (time.perf_counter() - started) * 1000.0
            finished_at = datetime.now(timezone.utc)
            metrics = ToolExecutionMetrics.from_latency(
                execution_time_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                started_at=started_at,
                finished_at=finished_at,
            )
            typed = ToolResult.ok(
                self.name,
                CalculatorResultData(result=result, expression=expression),
                metrics=metrics,
            )
            logger.info(
                "calculator completed: correlation_id=%s expression=%r result=%s",
                correlation_id,
                expression,
                result,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"expression": expression},
                output={
                    "result": result,
                    "expression": expression,
                    "typed_result": typed.model_dump(mode="json"),
                },
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                status=ToolExecutionStatus.SUCCESS,
                order=order,
                metrics=metrics,
                trace_metadata={"tool": self.name},
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - started) * 1000.0
            finished_at = datetime.now(timezone.utc)
            metrics = ToolExecutionMetrics.from_latency(
                execution_time_ms=latency_ms,
                status=ToolExecutionStatus.FAILED,
                failure_reason=str(exc),
                started_at=started_at,
                finished_at=finished_at,
            )
            typed = ToolResult[CalculatorResultData].fail(
                self.name,
                str(exc),
                metrics=metrics,
            )
            logger.error(
                "calculator failed: correlation_id=%s error=%s",
                correlation_id,
                exc,
            )
            return ToolExecution(
                tool_name=self.name,
                arguments={"expression": expression},
                output={"typed_result": typed.model_dump(mode="json")},
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
                status=ToolExecutionStatus.FAILED,
                error=str(exc),
                order=order,
                metrics=metrics,
                trace_metadata={"tool": self.name},
            )


__all__ = ["CalculatorTool", "evaluate_arithmetic"]
