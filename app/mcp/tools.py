"""MCP tool bridge — delegates to existing EmployeeService / RAGService.

No duplicated business logic. MCP handlers only validate inputs, resolve
trusted verification context, and call existing services.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import Context

from app.employees.service import EmployeeService
from app.mcp.context import (
    require_verified_employee_id,
    resolve_leave_confirmed,
)
from app.mcp.errors import McpToolExecutionError, safe_mcp_error
from app.mcp.schemas import McpToolResult
from app.orchestration.models import RAGRequest
from app.orchestration.rag_service import RAGService

logger = logging.getLogger(__name__)


class BuddieMcpToolBridge:
    """Service-backed implementations for Buddie MCP tools."""

    def __init__(
        self,
        *,
        employee_service: EmployeeService,
        rag_service: RAGService | None = None,
    ) -> None:
        self._employees = employee_service
        self._rag = rag_service

    def _ok(self, tool: str, data: dict[str, Any]) -> dict[str, Any]:
        return McpToolResult(ok=True, tool=tool, data=data).model_dump(mode="json")

    def _fail(self, tool: str, exc: BaseException) -> dict[str, Any]:
        err = safe_mcp_error(exc)
        logger.warning("MCP tool %s failed: code=%s", tool, err.code)
        return McpToolResult(
            ok=False,
            tool=tool,
            data={},
            error=err.message,
            error_code=err.code,
        ).model_dump(mode="json")

    def get_leave_balance(self, ctx: Context | None = None) -> dict[str, Any]:
        tool = "get_leave_balance"
        try:
            verified = require_verified_employee_id(ctx)
            return self._ok(tool, self._employees.get_leave_balance(verified))
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def get_leave_history(
        self,
        *,
        year: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        tool = "get_leave_history"
        try:
            verified = require_verified_employee_id(ctx)
            return self._ok(
                tool,
                self._employees.get_leave_history(verified, year=year),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def get_employee_profile(self, ctx: Context | None = None) -> dict[str, Any]:
        tool = "get_employee_profile"
        try:
            verified = require_verified_employee_id(ctx)
            return self._ok(tool, self._employees.get_employee_profile(verified))
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def get_manager_information(self, ctx: Context | None = None) -> dict[str, Any]:
        tool = "get_manager_information"
        try:
            verified = require_verified_employee_id(ctx)
            return self._ok(
                tool,
                self._employees.get_manager_information(verified),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def get_holiday_calendar(
        self,
        *,
        country: str | None = None,
        year: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        del ctx  # shared calendar — verification not required
        tool = "get_holiday_calendar"
        try:
            return self._ok(
                tool,
                self._employees.get_holiday_calendar(country=country, year=year),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def check_leave_eligibility(
        self,
        *,
        leave_type: str = "VACATION",
        requested_days: float = 1,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        tool = "check_leave_eligibility"
        try:
            verified = require_verified_employee_id(ctx)
            return self._ok(
                tool,
                self._employees.check_leave_eligibility(
                    verified,
                    leave_type=leave_type,
                    requested_days=requested_days,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def search_company_policy(
        self,
        *,
        query: str,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        del ctx
        tool = "search_company_policy"
        try:
            if self._rag is None:
                raise McpToolExecutionError(
                    "Policy search is unavailable in this MCP server configuration.",
                    code="rag_unavailable",
                )
            cleaned = (query or "").strip()
            if not cleaned:
                raise McpToolExecutionError(
                    "query is required for policy search.",
                    code="validation_error",
                )
            response = self._rag.query(RAGRequest(question=cleaned))
            docs_payload = [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "score": doc.score,
                    "metadata": dict(doc.metadata or {}),
                }
                for doc in response.retrieved_documents
            ]
            sources: list[str] = []
            for doc in response.retrieved_documents:
                meta = doc.metadata or {}
                label = meta.get("source") or meta.get("file_name") or doc.id
                if label:
                    sources.append(str(label))
            return self._ok(
                tool,
                {
                    "query": cleaned,
                    "summary": response.answer,
                    "answer": response.answer,
                    "documents": docs_payload,
                    "sources": sources,
                    "document_count": len(docs_payload),
                    "correlation_id": response.correlation_id,
                    "retrieval_metadata": dict(response.retrieval_metadata or {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)

    def create_leave_request(
        self,
        *,
        leave_type: str = "VACATION",
        start_date: str,
        end_date: str,
        reason: str = "Employee leave request",
        confirmed: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        tool = "create_leave_request"
        try:
            verified = require_verified_employee_id(ctx)
            allowed = resolve_leave_confirmed(
                confirmed_argument=bool(confirmed),
                ctx=ctx,
            )
            if not allowed:
                raise McpToolExecutionError(
                    "Leave request was not created. Explicit user confirmation "
                    "is required before write actions.",
                    code="confirmation_required",
                )
            return self._ok(
                tool,
                self._employees.create_leave_request(
                    verified,
                    leave_type=leave_type,
                    start_date=start_date,
                    end_date=end_date,
                    reason=reason,
                    confirmed=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(tool, exc)


__all__ = ["BuddieMcpToolBridge"]
