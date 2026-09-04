"""Structured failure classification and debug logging for Buddie evals.

Maps metric/agent/runtime outcomes to stable ``FailureKind`` values so logs
and JSON reports explain *why* a case failed without digging through nested
DeepEval objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Sequence

from evals.metrics.results import MetricScoreResult

logger = logging.getLogger(__name__)


class FailureKind(StrEnum):
    """Stable failure taxonomy for Buddie evaluation cases."""

    METRIC_THRESHOLD = "metric_threshold_failure"
    AGENT_CHECK = "agent_check_failure"
    SAFETY = "safety_failure"
    ROBUSTNESS = "robustness_failure"
    TOOL_WORKFLOW = "tool_workflow_failure"
    SEMANTIC_SIMILARITY = "semantic_similarity_failure"
    RUNTIME_TOOL_FAILURE = "runtime_tool_failure"
    RUNTIME_API_FAILURE = "runtime_api_failure"
    RUNTIME_EMPTY_RESPONSE = "runtime_empty_response"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    LLM_JUDGE_ERROR = "llm_judge_error"
    LLM_RATE_LIMITED = "llm_rate_limited"
    VERIFICATION_BYPASS = "verification_bypass"
    HITL_VIOLATION = "hitl_violation"
    PII_LEAKAGE = "pii_leakage"
    PROMPT_INJECTION_COMPLIANCE = "prompt_injection_compliance"
    UNAUTHORIZED_DATA_ACCESS = "unauthorized_data_access"
    UNWANTED_TOOL_CALL = "unwanted_tool_call"
    UNWANTED_RAG_ACTIVATION = "unwanted_rag_activation"
    GRACEFUL_DEGRADATION = "graceful_degradation_failure"


@dataclass(frozen=True)
class FailureDiagnostic:
    """One classified failure with a human debug hint."""

    kind: FailureKind
    message: str
    metric_name: str | None = None
    debug_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "message": self.message,
        }
        if self.metric_name:
            payload["metric_name"] = self.metric_name
        if self.debug_hint:
            payload["debug_hint"] = self.debug_hint
        return payload


@dataclass
class DiagnosticBundle:
    """Aggregated diagnostics for one golden case."""

    case_id: str
    diagnostics: list[FailureDiagnostic] = field(default_factory=list)

    def add(
        self,
        kind: FailureKind,
        message: str,
        *,
        metric_name: str | None = None,
        debug_hint: str = "",
    ) -> None:
        self.diagnostics.append(
            FailureDiagnostic(
                kind=kind,
                message=message,
                metric_name=metric_name,
                debug_hint=debug_hint,
            )
        )

    def failure_reasons(self) -> list[str]:
        return [f"{d.kind}: {d.message}" for d in self.diagnostics]

    def to_public_list(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.diagnostics]


_DEBUG_HINTS: dict[FailureKind, str] = {
    FailureKind.METRIC_THRESHOLD: (
        "Inspect retrieval_context vs expected_context; rerun with "
        "BUDDIE_EVAL_VERBOSE=1 for judge reasons."
    ),
    FailureKind.LLM_RATE_LIMITED: (
        "Gemini quota hit — retry later or set ENABLE_DEEPEVAL=false in CI "
        "and rely on deterministic metrics."
    ),
    FailureKind.LLM_JUDGE_ERROR: (
        "Check GOOGLE_API_KEY, judge model name, and DeepEval stack traces."
    ),
    FailureKind.INFRASTRUCTURE_ERROR: (
        "Agent collection threw — inspect correlation_id in agent logs."
    ),
    FailureKind.VERIFICATION_BYPASS: (
        "Session metadata may still carry verified_employee_id; check "
        "session_metadata_for_case and AgentService verification gate."
    ),
    FailureKind.HITL_VIOLATION: (
        "create_leave_request ran without confirmation — inspect planner "
        "and awaiting_confirmation metadata."
    ),
    FailureKind.PII_LEAKAGE: (
        "Answer leaked balances/IDs — check verification_status and tool "
        "outputs merged into final_answer."
    ),
    FailureKind.PROMPT_INJECTION_COMPLIANCE: (
        "Model complied with adversarial instruction — review system prompt "
        "and refusal routing for injection patterns."
    ),
    FailureKind.UNAUTHORIZED_DATA_ACCESS: (
        "Protected employee tool ran while unverified or for wrong employee."
    ),
    FailureKind.UNWANTED_TOOL_CALL: (
        "Planner routed to tools on a refuse/greeting case — check intent "
        "router and normalize_for_routing."
    ),
    FailureKind.UNWANTED_RAG_ACTIVATION: (
        "RAG ran when only structured tools were expected — check rag_used "
        "and selected_route metadata."
    ),
    FailureKind.RUNTIME_TOOL_FAILURE: (
        "Tool execution failed — inspect tools_invoked.error and MCP "
        "last_error in case metadata."
    ),
    FailureKind.RUNTIME_API_FAILURE: (
        "Upstream API/MCP transport failure — check MCP connectivity and "
        "employee store paths."
    ),
    FailureKind.GRACEFUL_DEGRADATION: (
        "Tool/API failed but answer exposed raw errors — check router "
        "_friendly_failure_message mapping."
    ),
    FailureKind.TOOL_WORKFLOW: (
        "Multi-tool order or success rate failed — compare tool_execution_order "
        "to expected_tools sequence."
    ),
    FailureKind.SEMANTIC_SIMILARITY: (
        "Token overlap below threshold — compare actual_output to "
        "expected_answer without LLM judge."
    ),
}


def _hint_for(kind: FailureKind) -> str:
    return _DEBUG_HINTS.get(kind, "")


def diagnose_metric_scores(
    case_id: str,
    scores: Sequence[MetricScoreResult],
) -> DiagnosticBundle:
    """Classify DeepEval / LLM metric outcomes."""
    bundle = DiagnosticBundle(case_id=case_id)
    for score in scores:
        if score.skipped:
            continue
        if score.rate_limited:
            bundle.add(
                FailureKind.LLM_RATE_LIMITED,
                f"{score.name}: {score.error or 'rate limited'}",
                metric_name=score.name,
                debug_hint=_hint_for(FailureKind.LLM_RATE_LIMITED),
            )
        elif score.error:
            bundle.add(
                FailureKind.LLM_JUDGE_ERROR,
                f"{score.name}: {score.error}",
                metric_name=score.name,
                debug_hint=_hint_for(FailureKind.LLM_JUDGE_ERROR),
            )
        elif score.passed is False:
            detail = score.reason or f"score={score.score} < {score.threshold}"
            bundle.add(
                FailureKind.METRIC_THRESHOLD,
                f"{score.name}: {detail}",
                metric_name=score.name,
                debug_hint=_hint_for(FailureKind.METRIC_THRESHOLD),
            )
    return bundle


def diagnose_deterministic_score(
    case_id: str,
    metric_name: str,
    score: float | None,
    *,
    kind: FailureKind,
    detail: str,
) -> DiagnosticBundle:
    """Record one deterministic metric failure (score < 1.0)."""
    bundle = DiagnosticBundle(case_id=case_id)
    if score is not None and score < 1.0:
        bundle.add(
            kind,
            f"{metric_name}: {detail} (score={score})",
            metric_name=metric_name,
            debug_hint=_hint_for(kind),
        )
    return bundle


def merge_diagnostic_bundles(
    case_id: str,
    bundles: Sequence[DiagnosticBundle],
) -> DiagnosticBundle:
    """Merge multiple bundles for the same case."""
    merged = DiagnosticBundle(case_id=case_id)
    for bundle in bundles:
        merged.diagnostics.extend(bundle.diagnostics)
    return merged


def log_case_diagnostics(
    case_id: str,
    diagnostics: Sequence[FailureDiagnostic],
    *,
    overall_status: str,
) -> None:
    """Emit structured log lines for post-run debugging."""
    if not diagnostics:
        if overall_status == "passed":
            logger.info(
                "eval_case_passed case_id=%s",
                case_id,
            )
        return
    for diag in diagnostics:
        logger.warning(
            "eval_case_failure case_id=%s status=%s kind=%s metric=%s "
            "message=%s debug_hint=%s",
            case_id,
            overall_status,
            diag.kind.value,
            diag.metric_name or "-",
            diag.message,
            diag.debug_hint or _hint_for(diag.kind),
        )


__all__ = [
    "DiagnosticBundle",
    "FailureDiagnostic",
    "FailureKind",
    "diagnose_deterministic_score",
    "diagnose_metric_scores",
    "log_case_diagnostics",
    "merge_diagnostic_bundles",
]
