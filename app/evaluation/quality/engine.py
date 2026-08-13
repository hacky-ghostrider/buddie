"""QualityGateEngine — turn rule results into PASS / WARNING / FAIL.

Composition: ``QualityGate`` (rules) + recommendations + decision policy.
The engine is the CI-facing entrypoint used by continuous evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config.settings import Settings, get_settings
from app.evaluation.quality.decision import QualityDecision, QualityStatus
from app.evaluation.quality.exceptions import QualityGateDisabledError
from app.evaluation.quality.gate import QualityGate
from app.evaluation.quality.models import (
    QualityGateThresholds,
    QualityRule,
    QualityRuleResult,
    RuleSeverity,
)
from app.evaluation.quality.recommendations import recommendations_for_results
from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class QualityGateEngine:
    """Apply quality gates and produce a ``QualityDecision``.

    Decision policy:
        * Any non-skipped FAIL-severity rule failure → ``FAIL``
        * Else any WARNING-severity failure, or overall score in
          ``[warning_threshold, pass_threshold)`` → ``WARNING``
        * Else → ``PASS``

    Args:
        gate: Optional preconfigured ``QualityGate``.
        thresholds: Optional thresholds (built from settings when omitted).
        settings: Application settings for defaults.
        allow_when_disabled: When False (default), raise if gates are off.
    """

    def __init__(
        self,
        *,
        gate: QualityGate | None = None,
        thresholds: QualityGateThresholds | None = None,
        settings: Settings | None = None,
        allow_when_disabled: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        self._thresholds = thresholds or QualityGateThresholds.from_settings(
            self._settings
        )
        self._gate = gate or QualityGate(thresholds=self._thresholds)
        self._allow_when_disabled = allow_when_disabled

    @property
    def thresholds(self) -> QualityGateThresholds:
        """Return active thresholds."""
        return self._thresholds

    @property
    def rules(self) -> list[QualityRule]:
        """Return active rules."""
        return self._gate.rules

    def evaluate(
        self,
        report: EvaluationReport,
        *,
        correlation_id: str | None = None,
        tool_failure_count: int | None = None,
        max_tool_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QualityDecision:
        """Run the quality gate and return a decision.

        Args:
            report: Evaluation report to gate.
            correlation_id: Optional correlation id.
            tool_failure_count: Optional tool failure override.
            max_tool_latency_ms: Optional max tool latency.
            total_latency_ms: Optional total latency.
            cost_usd: Optional cost.
            metadata: Optional decision metadata.

        Returns:
            ``QualityDecision`` with status PASS / WARNING / FAIL.

        Raises:
            QualityGateDisabledError: When gates are disabled and not allowed.
        """
        if not self._thresholds.enabled and not self._allow_when_disabled:
            raise QualityGateDisabledError(
                "Quality gates are disabled (QUALITY_GATE_ENABLED=false)"
            )

        if not self._thresholds.enabled:
            decision = QualityDecision(
                status=QualityStatus.PASS,
                reason="Quality gates disabled — auto PASS",
                failed_rules=[],
                warnings=[],
                overall_score=report.overall_score,
                correlation_id=correlation_id
                or (report.metadata or {}).get("correlation_id"),
                rule_results=[],
                recommendations=[],
                metadata={"quality_gate_enabled": False, **(metadata or {})},
            )
            logger.info(
                "Quality gate skipped (disabled): correlation_id=%s status=PASS",
                decision.correlation_id,
            )
            return decision

        rule_results = self._gate.evaluate(
            report,
            tool_failure_count=tool_failure_count,
            max_tool_latency_ms=max_tool_latency_ms,
            total_latency_ms=total_latency_ms,
            cost_usd=cost_usd,
        )
        decision = self._decide(
            report,
            rule_results,
            correlation_id=correlation_id,
            metadata=metadata,
        )
        logger.info(
            "Quality Decision: status=%s score=%.4f failed=%s warnings=%s "
            "correlation_id=%s reason=%r",
            decision.status.value,
            decision.overall_score,
            decision.failed_rules,
            decision.warnings,
            decision.correlation_id,
            decision.reason,
        )
        if decision.warnings:
            logger.warning(
                "Quality gate warnings: correlation_id=%s warnings=%s",
                decision.correlation_id,
                decision.warnings,
            )
        if decision.failed_rules:
            logger.error(
                "Quality gate failures: correlation_id=%s failed=%s",
                decision.correlation_id,
                decision.failed_rules,
            )
        return decision

    def evaluate_batch(
        self,
        reports: list[EvaluationReport],
        *,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> QualityDecision:
        """Evaluate each report and aggregate to a suite-level decision.

        Suite policy: any FAIL → FAIL; else any WARNING → WARNING; else PASS.
        Overall score is the mean of report overall scores.
        """
        if not reports:
            return QualityDecision(
                status=QualityStatus.FAIL,
                reason="No evaluation reports provided for quality gate",
                failed_rules=["empty_batch"],
                warnings=[],
                overall_score=0.0,
                correlation_id=correlation_id,
                metadata=dict(metadata or {}),
            )

        decisions = [
            self.evaluate(report, correlation_id=correlation_id, metadata=metadata)
            for report in reports
        ]
        mean_score = sum(d.overall_score for d in decisions) / len(decisions)
        failed: list[str] = []
        warnings: list[str] = []
        all_results: list[QualityRuleResult] = []
        for decision in decisions:
            failed.extend(decision.failed_rules)
            warnings.extend(decision.warnings)
            all_results.extend(decision.rule_results)

        if any(d.status == QualityStatus.FAIL for d in decisions):
            status = QualityStatus.FAIL
            reason = (
                f"Suite FAIL — {sum(1 for d in decisions if d.status == QualityStatus.FAIL)}"
                f"/{len(decisions)} reports failed quality gates"
            )
        elif any(d.status == QualityStatus.WARNING for d in decisions):
            status = QualityStatus.WARNING
            reason = (
                f"Suite WARNING — {sum(1 for d in decisions if d.status == QualityStatus.WARNING)}"
                f"/{len(decisions)} reports have warnings"
            )
        else:
            status = QualityStatus.PASS
            reason = f"Suite PASS — all {len(decisions)} reports passed quality gates"

        recommendations = recommendations_for_results(
            [r for r in all_results if not r.passed and not r.skipped]
        )
        aggregated = QualityDecision(
            status=status,
            reason=reason,
            failed_rules=sorted(set(failed)),
            warnings=sorted(set(warnings)),
            overall_score=round(mean_score, 6),
            correlation_id=correlation_id,
            rule_results=all_results,
            recommendations=recommendations,
            metadata={
                "batch_size": len(reports),
                "pass_count": sum(
                    1 for d in decisions if d.status == QualityStatus.PASS
                ),
                "warning_count": sum(
                    1 for d in decisions if d.status == QualityStatus.WARNING
                ),
                "fail_count": sum(
                    1 for d in decisions if d.status == QualityStatus.FAIL
                ),
                **(metadata or {}),
            },
        )
        logger.info(
            "Batch Quality Decision: status=%s reports=%d score=%.4f",
            aggregated.status.value,
            len(reports),
            aggregated.overall_score,
        )
        return aggregated

    def _decide(
        self,
        report: EvaluationReport,
        rule_results: list[QualityRuleResult],
        *,
        correlation_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> QualityDecision:
        """Apply PASS / WARNING / FAIL policy to rule results."""
        fail_ids = [
            r.rule_id.value
            for r in rule_results
            if not r.passed and not r.skipped and r.severity == RuleSeverity.FAIL
        ]
        warn_ids = [
            r.rule_id.value
            for r in rule_results
            if not r.passed and not r.skipped and r.severity == RuleSeverity.WARNING
        ]

        score = report.overall_score
        # Score band warning even when individual metric rules were skipped.
        score_warning = (
            self._thresholds.warning_threshold
            <= score
            < self._thresholds.pass_threshold
        )

        if fail_ids:
            status = QualityStatus.FAIL
            reason = f"FAIL — broken rules: {', '.join(fail_ids)}"
        elif warn_ids or score_warning:
            status = QualityStatus.WARNING
            parts: list[str] = []
            if warn_ids:
                parts.append(f"warned rules: {', '.join(warn_ids)}")
            if score_warning:
                parts.append(
                    f"overall score {score:.4f} below pass "
                    f"({self._thresholds.pass_threshold}) but above warning "
                    f"({self._thresholds.warning_threshold})"
                )
            reason = "WARNING — " + "; ".join(parts)
        else:
            status = QualityStatus.PASS
            reason = (
                f"PASS — all rules satisfied "
                f"(overall_score={score:.4f} >= {self._thresholds.pass_threshold})"
            )

        recommendations = recommendations_for_results(rule_results)
        corr = correlation_id or (report.metadata or {}).get("correlation_id")
        return QualityDecision(
            status=status,
            reason=reason,
            failed_rules=fail_ids,
            warnings=warn_ids,
            overall_score=score,
            correlation_id=str(corr) if corr else None,
            rule_results=rule_results,
            recommendations=recommendations,
            metadata={
                "question": report.question,
                "langsmith_run_url": report.langsmith_run_url,
                **(metadata or {}),
            },
        )


__all__ = ["QualityGateEngine"]
