"""Trusted verification context for MCP tool calls.

Protected employee tools must NOT trust arbitrary ``employee_id`` arguments.
The verified employee id is supplied only by the Buddie application via MCP
request ``meta`` (and optionally a same-process ContextVar for adapters).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from mcp.server.fastmcp import Context

from app.mcp.errors import McpToolExecutionError

META_VERIFIED_EMPLOYEE_ID = "buddie_verified_employee_id"
META_LEAVE_CONFIRMED = "buddie_leave_request_confirmed"
META_CORRELATION_ID = "buddie_correlation_id"

_trusted_verified_employee_id: ContextVar[str | None] = ContextVar(
    "buddie_mcp_verified_employee_id",
    default=None,
)
_trusted_leave_confirmed: ContextVar[bool] = ContextVar(
    "buddie_mcp_leave_confirmed",
    default=False,
)


def set_trusted_mcp_context(
    *,
    verified_employee_id: str | None = None,
    leave_request_confirmed: bool = False,
) -> tuple[Any, Any]:
    """Bind trusted values for the current async/sync call stack."""
    token_id = _trusted_verified_employee_id.set(
        str(verified_employee_id).strip().upper() if verified_employee_id else None
    )
    token_confirmed = _trusted_leave_confirmed.set(bool(leave_request_confirmed))
    return token_id, token_confirmed


def reset_trusted_mcp_context(tokens: tuple[Any, Any]) -> None:
    """Restore prior trusted context tokens."""
    token_id, token_confirmed = tokens
    _trusted_verified_employee_id.reset(token_id)
    _trusted_leave_confirmed.reset(token_confirmed)


def build_call_meta(
    *,
    verified_employee_id: str | None = None,
    leave_request_confirmed: bool = False,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Build MCP ``tools/call`` meta for the trusted application context."""
    meta: dict[str, Any] = {}
    if verified_employee_id:
        meta[META_VERIFIED_EMPLOYEE_ID] = str(verified_employee_id).strip().upper()
    if leave_request_confirmed:
        meta[META_LEAVE_CONFIRMED] = True
    if correlation_id:
        meta[META_CORRELATION_ID] = str(correlation_id)
    return meta


def _meta_value(ctx: Context | None, key: str) -> Any:
    if ctx is None:
        return None
    try:
        meta = ctx.request_context.meta
    except ValueError:
        return None
    if meta is None:
        return None
    extra = getattr(meta, "model_extra", None) or {}
    if key in extra:
        return extra[key]
    return getattr(meta, key, None)


def resolve_verified_employee_id(ctx: Context | None = None) -> str | None:
    """Resolve verified employee id from MCP meta or same-process ContextVar."""
    from_meta = _meta_value(ctx, META_VERIFIED_EMPLOYEE_ID)
    if from_meta:
        return str(from_meta).strip().upper()
    local = _trusted_verified_employee_id.get()
    if local:
        return str(local).strip().upper()
    return None


def require_verified_employee_id(ctx: Context | None = None) -> str:
    """Return verified employee id or raise a safe MCP error."""
    verified = resolve_verified_employee_id(ctx)
    if not verified:
        raise McpToolExecutionError(
            "Employee verification is required before accessing this data.",
            code="not_verified",
        )
    return verified


def resolve_leave_confirmed(
    *,
    confirmed_argument: bool,
    ctx: Context | None = None,
) -> bool:
    """Write ops require explicit confirmation from planner/HITL context."""
    if confirmed_argument:
        return True
    from_meta = _meta_value(ctx, META_LEAVE_CONFIRMED)
    if from_meta is True or str(from_meta).lower() in {"1", "true", "yes"}:
        return True
    return bool(_trusted_leave_confirmed.get())


__all__ = [
    "META_VERIFIED_EMPLOYEE_ID",
    "META_LEAVE_CONFIRMED",
    "META_CORRELATION_ID",
    "set_trusted_mcp_context",
    "reset_trusted_mcp_context",
    "build_call_meta",
    "resolve_verified_employee_id",
    "require_verified_employee_id",
    "resolve_leave_confirmed",
]
