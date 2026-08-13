"""MCP-facing error helpers — safe messages, no secrets / stack traces."""

from __future__ import annotations

from app.employees.exceptions import (
    EmployeeError,
    EmployeeNotVerifiedError,
    EmployeeValidationError,
    EmployeeVerificationError,
)


class McpToolExecutionError(Exception):
    """Raised when an MCP tool fails in a predictable, user-safe way."""

    def __init__(self, message: str, *, code: str = "tool_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def safe_mcp_error(exc: BaseException) -> McpToolExecutionError:
    """Map internal exceptions to non-sensitive MCP error messages."""
    if isinstance(exc, McpToolExecutionError):
        return exc
    if isinstance(exc, EmployeeNotVerifiedError):
        return McpToolExecutionError(
            "Employee verification is required before accessing this data.",
            code="not_verified",
        )
    if isinstance(exc, EmployeeVerificationError):
        return McpToolExecutionError(
            "Employee verification failed.",
            code="verification_failed",
        )
    if isinstance(exc, EmployeeValidationError):
        return McpToolExecutionError(str(exc), code="validation_error")
    if isinstance(exc, EmployeeError):
        return McpToolExecutionError(
            "Unable to complete the employee data request.",
            code="employee_error",
        )
    return McpToolExecutionError(
        "The tool could not complete this request.",
        code="tool_error",
    )


__all__ = ["McpToolExecutionError", "safe_mcp_error"]
