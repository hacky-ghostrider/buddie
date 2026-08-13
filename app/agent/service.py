"""AgentService — application façade over the LangGraph agent.

Orchestrates:
    question → graph (planner → router → finalize) → EvaluationContext
             → ToolValidator → LangSmith trace → AgentRunResult

Dependency injection keeps RAG, tracing, and validation swappable for tests.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.agent.conversation import normalize_for_routing, normalize_message
from app.agent.graph import build_agent_graph
from app.agent.models import AgentRunResult, PlannerOutput
from app.agent.planner import Planner
from app.agent.router import ToolRouter, executions_from_state
from app.agent.state import AgentState, empty_agent_state
from app.agent.tools import ToolRegistry, build_default_tool_registry
from app.config.settings import Settings, get_settings
from app.employees.service import EmployeeService
from app.evaluation.context import EvaluationContext
from app.evaluation.tool_validation.tool_contract import ToolContract
from app.evaluation.tool_validation.tool_execution import ToolExecution
from app.evaluation.tool_validation.trace_mapper import ToolTraceMapper
from app.evaluation.tool_validation.validator import ToolValidator
from app.mcp.client import BuddieMcpClient
from app.orchestration.models import RAGResponse
from app.orchestration.rag_service import RAGService
from app.tracing.base import TraceRecord, TraceSpanData
from app.tracing.service import TracingService, create_tracer

logger = logging.getLogger(__name__)

# Keys that must never be echoed into Developer Mode / API metadata.
_SENSITIVE_META_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "openai_api_key",
    "langsmith_api_key",
    "bearer",
}


class AgentService:
    """Run the production LangGraph agent with evaluation integration.

    Args:
        rag_service: RAG orchestrator reused by ``search_docs`` / ``summarize``.
        registry: Optional tool registry (defaults built from ``rag_service``).
        planner: Optional planner override.
        tracing_service: Optional tracing façade.
        tool_validator: Optional tool validator.
        settings: Application settings.
    """

    def __init__(
        self,
        *,
        rag_service: RAGService,
        employee_service: EmployeeService | None = None,
        registry: ToolRegistry | None = None,
        planner: Planner | None = None,
        tracing_service: TracingService | None = None,
        tool_validator: ToolValidator | None = None,
        settings: Settings | None = None,
        mcp_client: BuddieMcpClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._rag_service = rag_service
        self._employee_service = employee_service
        self._mcp_client: BuddieMcpClient | None = mcp_client
        if registry is not None:
            self._registry = registry
        else:
            self._registry, self._mcp_client = self._build_registry(
                rag_service,
                employee_service=employee_service,
            )
        self._planner = planner or Planner()
        self._router = ToolRouter(self._registry)
        self._graph = build_agent_graph(planner=self._planner, router=self._router)
        self._tracing_service = tracing_service or TracingService(
            tracer=create_tracer(self._settings),
            settings=self._settings,
        )
        self._tool_validator = tool_validator or ToolValidator()
        self._trace_mapper = ToolTraceMapper()

    def _build_registry(
        self,
        rag_service: RAGService,
        *,
        employee_service: EmployeeService | None,
    ) -> tuple[ToolRegistry, BuddieMcpClient | None]:
        """Build Direct or MCP-backed registry from settings."""
        mode = str(getattr(self._settings, "buddie_tool_mode", "direct") or "direct")
        if mode != "mcp":
            return (
                build_default_tool_registry(
                    rag_service,
                    employee_service=employee_service,
                ),
                None,
            )
        from app.mcp.adapter import build_tool_registry

        command = str(getattr(self._settings, "mcp_server_command", "") or "").strip()
        command_parts = command.split() if command else None
        return build_tool_registry(
            rag_service,
            employee_service=employee_service,
            tool_mode="mcp",
            mcp_transport=str(getattr(self._settings, "mcp_transport", "memory")),
            mcp_server_command=command_parts,
            mcp_server_url=str(getattr(self._settings, "mcp_server_url", "") or "")
            or None,
            mcp_timeout_seconds=float(
                getattr(self._settings, "mcp_timeout_seconds", 30.0)
            ),
            mcp_client=self._mcp_client,
        )

    @property
    def registry(self) -> ToolRegistry:
        """Expose the tool registry for tests / introspection."""
        return self._registry

    @property
    def mcp_client(self) -> BuddieMcpClient | None:
        """Expose the MCP client when tool mode is MCP."""
        return self._mcp_client

    @property
    def graph(self) -> Any:
        """Expose the compiled LangGraph runnable."""
        return self._graph

    def run(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
        expected_answer: str | None = None,
        expected_sources: list[str] | None = None,
        validate_tools: bool = True,
        correlation_id: str | None = None,
    ) -> AgentRunResult:
        """Execute one agent turn end-to-end.

        Args:
            question: User question.
            metadata: Optional run metadata (e.g. scenario id).
            expected_answer: Optional golden answer for EvaluationContext.
            expected_sources: Optional expected sources.
            validate_tools: When True, run ``ToolValidator`` on contracts.
            correlation_id: Optional correlation id (generated when omitted).

        Returns:
            ``AgentRunResult`` with executions, context, and optional validation.
        """
        started = time.perf_counter()
        corr = correlation_id or str(uuid.uuid4())
        meta = dict(metadata or {})
        cleaned_question = (question or "").strip()
        initial = empty_agent_state(
            question=cleaned_question,
            correlation_id=corr,
            metadata=meta,
        )

        logger.info(
            "Agent run started: correlation_id=%s question_preview=%r",
            corr,
            cleaned_question[:80],
        )

        final_state: AgentState = self._graph.invoke(initial)
        executions = executions_from_state(final_state)
        planner_output = None
        planner_decision = None
        if final_state.get("planner_output"):
            planner_output = PlannerOutput.model_validate(
                final_state["planner_output"]
            )
            planner_decision = planner_output.to_decision()

        evaluation_context = self._build_evaluation_context(
            final_state,
            executions=executions,
            expected_answer=expected_answer,
            expected_sources=expected_sources,
        )

        tool_validation = None
        contracts = (
            list(planner_output.tool_contracts)
            if planner_output is not None
            else [
                ToolContract.model_validate(item)
                for item in (final_state.get("tool_contracts") or [])
            ]
        )
        if validate_tools and self._settings.enable_tool_validation and contracts:
            tool_validation = self._tool_validator.validate_contracts(
                contracts,
                executions,
                metadata={"correlation_id": corr, **meta},
            )

        # LangSmith-like payload for ToolTraceMapper path (never parsed by validator).
        langsmith_payload = self._executions_to_langsmith_payload(
            executions,
            planner_output=planner_output,
            correlation_id=corr,
        )
        # Prove mapper round-trip stays available for automation consumers.
        # Do not replace a legitimate empty tool list (conversational / gate
        # routes) with planner-only mapper output.
        mapped = self._trace_mapper.map_langsmith_trace(langsmith_payload)
        if mapped and not executions:
            tool_only = [e for e in mapped if e.tool_name != "planner"]
            if tool_only:
                executions = mapped

        rag_snapshot = (final_state.get("metadata") or {}).get("last_rag_response")
        trace = self._record_trace(
            question=cleaned_question,
            final_answer=final_state.get("final_answer") or "",
            executions=executions,
            evaluation_context=evaluation_context,
            planner_output=planner_output,
            tool_validation=(
                tool_validation.to_summary_dict() if tool_validation else None
            ),
            correlation_id=corr,
            langsmith_payload=langsmith_payload,
            rag_snapshot=rag_snapshot if isinstance(rag_snapshot, dict) else None,
        )

        # Attach LangSmith ids onto evaluation context.
        evaluation_context = evaluation_context.model_copy(
            update={
                "langsmith_trace_id": trace.trace_id,
                "langsmith_run_id": trace.run_id,
                "langsmith_run_url": trace.run_url,
            }
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        state_meta = dict(final_state.get("metadata") or {})
        intent_route = (
            state_meta.get("intent_route")
            or (planner_output.intent_route if planner_output else None)
        )
        selected_tools = list(final_state.get("selected_tools") or [])
        if not selected_tools:
            selected_tools = [e.tool_name for e in executions]
        # Surface MULTI_TOOL for genuine multi-capability turns.
        # Keep retrieve-then-summarize as the knowledge route (not MULTI_TOOL).
        rag_pipeline_only = set(selected_tools) <= {"search_docs", "summarize"} and (
            len(selected_tools) >= 1
        )
        selected_route = (
            "MULTI_TOOL"
            if len(selected_tools) > 1 and not rag_pipeline_only
            else intent_route
        )
        verified_id = (
            state_meta.get("verified_employee_id")
            or meta.get("verified_employee_id")
            or meta.get("employee_id")
        )
        rag_used = bool(
            state_meta.get("last_rag_response")
            or any(
                e.tool_name in {"search_docs", "summarize", "search_company_policy"}
                for e in executions
            )
            or (evaluation_context.retrieved_documents if evaluation_context else None)
            or (evaluation_context.retrieved_chunks if evaluation_context else None)
        )
        retrieved_sources: list[str] = []
        for execution in executions:
            if not execution.success or not isinstance(execution.output, dict):
                continue
            if execution.tool_name in {
                "search_docs",
                "search_company_policy",
                "summarize",
            }:
                sources = execution.output.get("sources")
                if isinstance(sources, list):
                    retrieved_sources.extend(str(s) for s in sources if s)
                docs = execution.output.get("documents")
                if isinstance(docs, list):
                    for doc in docs:
                        if isinstance(doc, dict):
                            meta_doc = doc.get("metadata") or {}
                            label = (
                                meta_doc.get("source")
                                or meta_doc.get("file_name")
                                or doc.get("id")
                            )
                            if label:
                                retrieved_sources.append(str(label))
        # Deduplicate while preserving order.
        seen_sources: set[str] = set()
        unique_sources: list[str] = []
        for source in retrieved_sources:
            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)

        pending_leave = state_meta.get("pending_leave_request")
        mcp_snapshot = self._mcp_metadata(executions)
        result = AgentRunResult(
            question=cleaned_question,
            final_answer=final_state.get("final_answer") or "",
            tool_executions=executions,
            planner_output=planner_output,
            planner_decision=planner_decision,
            evaluation_context=evaluation_context,
            tool_validation=tool_validation,
            correlation_id=corr,
            trace_id=trace.trace_id,
            run_id=trace.run_id,
            run_url=trace.run_url,
            latency_ms=latency_ms,
            metadata={
                **_safe_public_metadata(meta),
                "original_input": cleaned_question,
                "normalized_input": normalize_message(cleaned_question),
                "routing_normalized_input": normalize_for_routing(cleaned_question),
                "detected_intent": intent_route,
                "selected_route": selected_route,
                "selected_tools": selected_tools,
                "intent_route": intent_route,
                "verification_status": (
                    "verified" if verified_id else "unverified"
                ),
                "verified_employee_id": (
                    str(verified_id).strip().upper() if verified_id else None
                ),
                "rag_used": rag_used,
                "retrieved_sources": unique_sources,
                "tool_execution_order": [
                    e.tool_name for e in executions
                ],
                "tools_invoked": [
                    {
                        "tool_name": e.tool_name,
                        "status": (
                            e.status.value
                            if hasattr(e.status, "value")
                            else str(e.status)
                        ),
                        "arguments": _safe_tool_value(e.arguments),
                        "result_summary": _summarize_tool_output(e.output),
                        "latency_ms": e.latency_ms,
                        "error": (
                            "tool_error"
                            if (not e.success and e.error)
                            else None
                        ),
                        "order": e.order,
                        "protocol": (
                            (e.trace_metadata or {}).get("protocol")
                            or ("MCP" if mcp_snapshot.get("used") else "direct")
                        ),
                        "mcp_latency_ms": (e.trace_metadata or {}).get(
                            "mcp_latency_ms"
                        ),
                    }
                    for e in executions
                ],
                "pending_leave_request": pending_leave,
                "awaiting_confirmation": bool(
                    state_meta.get("awaiting_confirmation")
                ),
                "human_confirmation_required": bool(
                    state_meta.get("awaiting_confirmation")
                ),
                "final_result": (final_state.get("final_answer") or "")[:500],
                "latency_ms": latency_ms,
                "langsmith_run_url": trace.run_url,
                "trace_enabled": trace.enabled,
                **mcp_snapshot,
            },
        )
        logger.info(
            "Agent run completed: correlation_id=%s tools=%s passed=%s latency_ms=%.1f",
            corr,
            [e.tool_name for e in executions],
            None if tool_validation is None else tool_validation.passed,
            latency_ms,
        )
        return result

    def _build_evaluation_context(
        self,
        state: AgentState,
        *,
        executions: list[ToolExecution],
        expected_answer: str | None,
        expected_sources: list[str] | None,
    ) -> EvaluationContext:
        """Prefer finalize-node context; enrich with expected fields."""
        raw = state.get("evaluation_context")
        if raw:
            payload = dict(raw)
            payload.pop("generated_answer", None)
            context = EvaluationContext.model_validate(payload)
            return context.model_copy(
                update={
                    "expected_answer": expected_answer,
                    "expected_sources": expected_sources,
                    "tool_calls": executions,
                    "tool_results": [e.output for e in executions],
                    "answer": state.get("final_answer") or context.answer,
                }
            )
        return EvaluationContext(
            question=state.get("question", ""),
            original_user_request=state.get("question", ""),
            tool_calls=executions,
            tool_results=[e.output for e in executions],
            answer=state.get("final_answer") or "",
            expected_answer=expected_answer,
            expected_sources=expected_sources,
            correlation_id=state.get("correlation_id"),
            latency_ms=sum(e.latency_ms or 0.0 for e in executions),
            metadata=dict(state.get("metadata") or {}),
        )

    def _mcp_metadata(self, executions: list[ToolExecution]) -> dict[str, Any]:
        """Build safe MCP observability fields for Developer / Evaluation mode."""
        mcp_tools = [
            e
            for e in executions
            if (e.trace_metadata or {}).get("protocol") == "MCP"
        ]
        used = bool(mcp_tools) or self._mcp_client is not None
        if not used:
            return {
                "tool_mode": str(
                    getattr(self._settings, "buddie_tool_mode", "direct") or "direct"
                ),
                "mcp": {
                    "used": False,
                    "connected": False,
                    "protocol": None,
                },
            }

        client_status = (
            self._mcp_client.status_snapshot() if self._mcp_client is not None else {}
        )
        mcp_latencies = [
            float((e.trace_metadata or {}).get("mcp_latency_ms") or e.latency_ms or 0.0)
            for e in mcp_tools
        ]
        connected = bool(client_status.get("connected")) or bool(mcp_tools)
        unauthorized: list[str] = []
        for e in mcp_tools:
            if e.success:
                continue
            code = str((e.trace_metadata or {}).get("error_code") or "").lower()
            err = str(e.error or "").lower()
            if (
                code in {"not_verified", "verification_failed", "confirmation_required"}
                or "verif" in err
                or "confirmation" in err
            ):
                unauthorized.append(e.tool_name)
        return {
            "tool_mode": "mcp",
            "mcp": {
                "used": True,
                "connected": connected,
                "protocol": "MCP",
                "transport": client_status.get("transport")
                or (mcp_tools[0].trace_metadata or {}).get("mcp_transport")
                if mcp_tools
                else None,
                "server_status": "connected" if connected else "unavailable",
                "tool_count": client_status.get("tool_count"),
                "discovered_tools": client_status.get("tools"),
                "mcp_latency_ms": round(sum(mcp_latencies), 3) if mcp_latencies else (
                    client_status.get("mcp_latency_ms")
                ),
                "tool_latency_ms": [
                    {
                        "tool_name": e.tool_name,
                        "latency_ms": e.latency_ms,
                        "mcp_latency_ms": (e.trace_metadata or {}).get(
                            "mcp_latency_ms"
                        ),
                        "status": (
                            e.status.value
                            if hasattr(e.status, "value")
                            else str(e.status)
                        ),
                        "order": e.order,
                    }
                    for e in mcp_tools
                ],
                "unauthorized_attempts": unauthorized,
                "last_error": client_status.get("last_error"),
            },
        }

    def _record_trace(
        self,
        *,
        question: str,
        final_answer: str,
        executions: list[ToolExecution],
        evaluation_context: EvaluationContext,
        planner_output: PlannerOutput | None,
        tool_validation: dict[str, Any] | None,
        correlation_id: str,
        langsmith_payload: dict[str, Any],
        rag_snapshot: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """Record planner + tool + answer span via ``TracingService``."""
        # Prefer TracingService.trace_rag_evaluation when RAG snapshot exists.
        if isinstance(rag_snapshot, dict) and rag_snapshot.get("answer") is not None:
            try:
                rag_response = RAGResponse.model_validate(rag_snapshot)
                return self._tracing_service.trace_rag_evaluation(
                    rag_response=rag_response,
                    tool_validation=tool_validation,
                    prompt=evaluation_context.prompt,
                    extra_metadata={
                        "correlation_id": correlation_id,
                        "agent": True,
                        "planner_output": (
                            planner_output.model_dump(mode="json")
                            if planner_output
                            else None
                        ),
                        "selected_tools": [e.tool_name for e in executions],
                        "tool_contracts": (
                            [c.model_dump(mode="json") for c in planner_output.tool_contracts]
                            if planner_output
                            else []
                        ),
                        "tool_executions": [
                            e.model_dump(mode="json") for e in executions
                        ],
                        "langsmith_child_runs": langsmith_payload,
                        "final_answer": final_answer,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Falling back to generic agent trace: %s",
                    exc,
                )

        span = TraceSpanData(
            question=question,
            retrieved_chunks=list(evaluation_context.retrieved_chunks),
            prompt=evaluation_context.prompt,
            model=evaluation_context.model,
            tokens=dict(evaluation_context.token_usage),
            latency_ms=evaluation_context.latency_ms,
            answer=final_answer,
            evaluation_results={
                "tool_validation": tool_validation,
                "planner": (
                    planner_output.model_dump(mode="json") if planner_output else None
                ),
            },
            metadata={
                "correlation_id": correlation_id,
                "agent": True,
                "selected_tools": [e.tool_name for e in executions],
                "tool_contracts": (
                    [c.model_dump(mode="json") for c in planner_output.tool_contracts]
                    if planner_output
                    else []
                ),
                "tool_executions": [e.model_dump(mode="json") for e in executions],
                "langsmith_child_runs": langsmith_payload,
            },
        )
        return self._tracing_service.tracer.record(span)

    @staticmethod
    def _executions_to_langsmith_payload(
        executions: list[ToolExecution],
        *,
        planner_output: PlannerOutput | None,
        correlation_id: str,
    ) -> dict[str, Any]:
        """Build a LangSmith-like dict for ``ToolTraceMapper`` (no SDK parse)."""
        child_runs: list[dict[str, Any]] = []
        if planner_output is not None:
            child_runs.append(
                {
                    "name": "planner",
                    "run_type": "chain",
                    "inputs": {"rationale": planner_output.rationale},
                    "outputs": {
                        "selected_tools": planner_output.selected_tools,
                        "tool_contracts": [
                            c.model_dump(mode="json")
                            for c in planner_output.tool_contracts
                        ],
                    },
                    "status": "success",
                    "metadata": {"correlation_id": correlation_id},
                }
            )
        for execution in executions:
            child_runs.append(
                {
                    "name": execution.tool_name,
                    "tool_name": execution.tool_name,
                    "run_type": "tool",
                    "inputs": dict(execution.arguments),
                    "arguments": dict(execution.arguments),
                    "outputs": execution.output,
                    "output": execution.output,
                    "latency_ms": execution.latency_ms,
                    "status": execution.status.value,
                    "error": execution.error,
                    "retry_count": execution.retry_count,
                    "order": execution.order,
                    "started_at": (
                        execution.started_at.isoformat()
                        if execution.started_at
                        else None
                    ),
                    "finished_at": (
                        execution.finished_at.isoformat()
                        if execution.finished_at
                        else None
                    ),
                    "metadata": dict(execution.trace_metadata),
                }
            )
        return {
            "child_runs": child_runs,
            "correlation_id": correlation_id,
        }


def _safe_public_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Copy inbound metadata while dropping secrets / credential-like keys."""
    safe: dict[str, Any] = {}
    for key, value in meta.items():
        lowered = str(key).strip().lower()
        if lowered in _SENSITIVE_META_KEYS or any(
            token in lowered for token in ("api_key", "token", "secret", "password")
        ):
            continue
        # Avoid echoing long prior answers / raw histories into every response.
        if lowered in {
            "last_assistant_message",
            "prior_assistant_message",
            "last_answer",
            "conversation_history",
            "messages",
        }:
            continue
        safe[key] = value
    return safe


def _safe_tool_value(value: Any) -> Any:
    """Recursively redact credential-like keys from tool args/results."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _SENSITIVE_META_KEYS or any(
                token in lowered for token in ("api_key", "token", "secret", "password")
            ):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = _safe_tool_value(item)
        return cleaned
    if isinstance(value, list):
        return [_safe_tool_value(item) for item in value[:20]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value


def _summarize_tool_output(output: Any) -> str:
    """Short developer-facing summary of a tool result (no raw dumps)."""
    if output is None:
        return "empty"
    if isinstance(output, str):
        text = output.strip()
        return text[:180] + ("…" if len(text) > 180 else "")
    if isinstance(output, dict):
        if "leave_history" in output:
            items = output.get("leave_history") or []
            return (
                f"leave_history entries={len(items)} "
                f"total_days={output.get('total_days')}"
            )
        if "leave_balance" in output:
            lb = output.get("leave_balance") or {}
            return (
                "leave_balance "
                f"vacation={lb.get('vacation')} sick={lb.get('sick')} "
                f"personal={lb.get('personal')}"
            )
        if "eligible" in output and "leave_type" in output:
            return (
                f"eligibility leave_type={output.get('leave_type')} "
                f"eligible={output.get('eligible')}"
            )
        if output.get("created") and output.get("request_id"):
            return f"leave_request created id={output.get('request_id')}"
        if "manager" in output and "employee_name" in output:
            return f"manager={output.get('manager')}"
        if "holidays" in output:
            return f"holiday_calendar count={output.get('count', 0)}"
        if "pending_actions" in output:
            return f"pending_actions count={output.get('count', 0)}"
        if "upcoming_holidays" in output or "next_holiday" in output:
            nxt = output.get("next_holiday") or {}
            return f"next_holiday={nxt.get('holiday_name') or 'none'}"
        if "payroll" in output:
            return "payroll summary available"
        if output.get("verified") is True:
            return f"verified employee_id={output.get('employee_id')}"
        if "summary" in output and output["summary"] is not None:
            summary = str(output["summary"])
            return summary[:180] + ("…" if len(summary) > 180 else "")
        if "results" in output:
            results = output.get("results") or []
            return f"results count={len(results) if isinstance(results, list) else 1}"
        keys = ", ".join(list(output.keys())[:6])
        return f"dict keys=[{keys}]"
    if isinstance(output, list):
        return f"list length={len(output)}"
    return type(output).__name__


__all__ = ["AgentService"]
