"""Collect DeepEval-ready fields by executing Buddie goldens via AgentService.

Sprint 18 seam:
    BuddieGoldenCase → AgentService.run → DeepEvalCompatibleCase

retrieval_context uses runtime evidence only (never golden expected_context).
Metric suite lives in ``evals.runners.deepeval_suite``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from app.agent.models import AgentRunResult
from app.evaluation.tool_validation.tool_execution import ToolExecution
from evals.golden_dataset.models import BuddieGoldenCase, BuddieGoldenDataset
from evals.runners.deepeval_case import DeepEvalCompatibleCase

logger = logging.getLogger(__name__)

# Tool names whose evidence is usually already present as RAG retrieved chunks.
_RAG_TOOL_NAMES = frozenset({"search_docs", "summarize", "search_company_policy"})


class AgentRunner(Protocol):
    """Minimal protocol matching ``AgentService.run`` for DI / tests."""

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
        """Execute one agent turn."""


def session_metadata_for_case(
    dataset: BuddieGoldenDataset,
    case: BuddieGoldenCase,
) -> dict[str, Any]:
    """Build AgentService metadata for one golden case.

    Default session uses ``default_session.verified_employee_id`` (E-1101).
    Verification-negative cases intentionally omit verified identity.
    """
    meta: dict[str, Any] = {
        "evaluation_case_id": case.id,
        "evaluation_category": case.category,
        "evaluation_behavior": case.expected_behavior,
    }
    if case.expected_behavior == "require_verification":
        return meta

    employee_id = dataset.default_session.get("verified_employee_id")
    if isinstance(employee_id, str) and employee_id.strip():
        meta["verified_employee_id"] = employee_id.strip().upper()
    return meta


def _tool_output_as_context(tool_name: str, output: Any) -> str | None:
    """Serialize a successful tool output into a retrieval_context string."""
    if output is None:
        return None
    if isinstance(output, str):
        text = output.strip()
        return f"{tool_name}: {text}" if text else None
    try:
        payload = json.dumps(output, default=str, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(output)
    payload = payload.strip()
    if not payload or payload in {"{}", "[]", "null"}:
        return None
    return f"{tool_name}: {payload}"


def build_retrieval_context(
    result: AgentRunResult,
    case: BuddieGoldenCase | None = None,
) -> list[str]:
    """Assemble DeepEval ``retrieval_context`` from runtime evidence only.

    Preference order:
    1. RAG retrieved chunk texts from ``EvaluationContext``
    2. Successful tool outputs (JSON / text), including employee tools

    Golden ``expected_context`` is reference/ground-truth data and is **never**
    injected here. When runtime produces no evidence, return ``[]``.

    ``case`` is accepted for call-site compatibility but is unused for evidence.
    """
    _ = case  # never substitute golden expected_context into retrieval_context
    texts: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        if value is None:
            return
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        texts.append(cleaned)

    ctx = result.evaluation_context
    if ctx is not None:
        for chunk in ctx.retrieved_chunks:
            _add(chunk)
        if not texts:
            for doc in ctx.retrieved_documents:
                _add(doc.text)

    rag_chunks_present = bool(texts)
    for execution in result.tool_executions:
        if not isinstance(execution, ToolExecution):
            continue
        if not execution.success:
            continue
        # Avoid duplicating RAG document bodies already captured from snapshot.
        if rag_chunks_present and execution.tool_name in _RAG_TOOL_NAMES:
            continue
        _add(_tool_output_as_context(execution.tool_name, execution.output))

    return texts


def collect_deepeval_case(
    dataset: BuddieGoldenDataset,
    case: BuddieGoldenCase,
    agent: AgentRunner,
    *,
    validate_tools: bool = False,
) -> DeepEvalCompatibleCase:
    """Run one golden case through Buddie runtime and map to DeepEval fields.

    Args:
        dataset: Full golden dataset (provides default session).
        case: Single golden case.
        agent: ``AgentService`` or compatible runner.
        validate_tools: Forwarded to ``AgentService.run`` (off by default so
            collection stays focused on DeepEval field capture).

    Returns:
        ``DeepEvalCompatibleCase`` ready for later metric adapters.
    """
    metadata = session_metadata_for_case(dataset, case)
    logger.info(
        "Collecting DeepEval case: id=%s behavior=%s verified=%s",
        case.id,
        case.expected_behavior,
        bool(metadata.get("verified_employee_id")),
    )
    result = agent.run(
        case.user_query,
        metadata=metadata,
        expected_answer=case.expected_answer,
        validate_tools=validate_tools,
        correlation_id=f"eval-{case.id}",
    )
    actual = (result.final_answer or "").strip()
    if not actual:
        # Keep DeepEval cases constructible even on empty agent turns.
        actual = "(empty)"

    retrieval_context = build_retrieval_context(result, case)
    tool_names = [e.tool_name for e in result.tool_executions]
    meta = result.metadata or {}
    tools_invoked = meta.get("tools_invoked")
    if not isinstance(tools_invoked, list):
        tools_invoked = [
            {
                "tool_name": e.tool_name,
                "arguments": dict(e.arguments or {}),
                "success": bool(e.success),
            }
            for e in result.tool_executions
        ]
    return DeepEvalCompatibleCase(
        case_id=case.id,
        input=case.user_query,
        actual_output=actual,
        expected_output=case.expected_answer,
        retrieval_context=retrieval_context,
        context=list(retrieval_context),
        category=case.category,
        expected_behavior=case.expected_behavior,
        expected_context=list(case.expected_context),
        metadata={
            # ACTUAL runtime evidence / agent observations (not golden annotations)
            "correlation_id": result.correlation_id,
            "selected_tools": list(meta.get("selected_tools") or tool_names),
            "tool_execution_order": tool_names,
            "tools_invoked": tools_invoked,
            "verification_status": meta.get("verification_status"),
            "verified_employee_id": meta.get("verified_employee_id"),
            "awaiting_confirmation": bool(meta.get("awaiting_confirmation")),
            "human_confirmation_required": bool(
                meta.get("human_confirmation_required")
            ),
            "rag_used": meta.get("rag_used"),
            "latency_ms": result.latency_ms,
            # Annotated expectations echoed for agent checks (reference only)
            "expected_tool": case.expected_tool,
            "expected_tools": list(case.expected_tools),
        },
    )


def collect_all_deepeval_cases(
    dataset: BuddieGoldenDataset,
    agent: AgentRunner,
    *,
    validate_tools: bool = False,
) -> list[DeepEvalCompatibleCase]:
    """Execute every golden case and return DeepEval-compatible cases."""
    return [
        collect_deepeval_case(
            dataset,
            case,
            agent,
            validate_tools=validate_tools,
        )
        for case in dataset.cases
    ]


__all__ = [
    "AgentRunner",
    "build_retrieval_context",
    "collect_all_deepeval_cases",
    "collect_deepeval_case",
    "session_metadata_for_case",
]
