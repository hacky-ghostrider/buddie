"""QualityGate — apply configured rules to an EvaluationReport.

The gate is a pure policy evaluator: it does not run DeepEval, LangSmith,
or agents. It receives an ``EvaluationReport`` (and optional signals) and
emits per-rule results for the engine to turn into a ``QualityDecision``.
"""

from __future__ import annotations

import logging

from app.evaluation.quality.exceptions import InvalidQualityRuleError
from app.evaluation.quality.models import (
    QualityGateThresholds,
    QualityRule,
    QualityRuleId,
    QualityRuleResult,
    RuleOperator,
    build_default_rules,
)
from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class QualityGate:
    """Evaluate an ``EvaluationReport`` against a list of ``QualityRule``s.

    Args:
        rules: Explicit rules. When omitted, built from ``thresholds``.
        thresholds: Threshold config (used when ``rules`` is omitted and for
            missing-metric skip behaviour).
    """

    def __init__(
        self,
        *,
        rules: list[QualityRule] | None = None,
        thresholds: QualityGateThresholds | None = None,
    ) -> None:
        self._thresholds = thresholds or QualityGateThresholds()
        self._rules = list(rules) if rules is not None else build_default_rules(
            self._thresholds
        )
        if not self._rules:
            raise InvalidQualityRuleError("QualityGate requires at least one rule")

    @property
    def rules(self) -> list[QualityRule]:
        """Return configured rules (copy)."""
        return list(self._rules)

    @property
    def thresholds(self) -> QualityGateThresholds:
        """Return threshold configuration."""
        return self._thresholds

    def evaluate(
        self,
        report: EvaluationReport,
        *,
        tool_failure_count: int | None = None,
        max_tool_latency_ms: float | None = None,
        total_latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> list[QualityRuleResult]:
        """Apply every enabled rule and return ordered results.

        Args:
            report: Evaluation report under gate review.
            tool_failure_count: Optional override for tool failure count.
            max_tool_latency_ms: Optional max observed tool latency.
            total_latency_ms: Optional end-to-end latency override.
            cost_usd: Optional cost override.

        Returns:
            Ordered ``QualityRuleResult`` list.
        """
        signals = self._resolve_signals(
            report,
            tool_failure_count=tool_failure_count,
            max_tool_latency_ms=max_tool_latency_ms,
            total_latency_ms=total_latency_ms,
            cost_usd=cost_usd,
        )
        results: list[QualityRuleResult] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            results.append(self._evaluate_rule(rule, report, signals))
        logger.info(
            "QualityGate evaluated: question=%r rules=%d failed=%d warned=%d",
            report.question[:80],
            len(results),
            sum(1 for r in results if not r.passed and not r.skipped),
            sum(
                1
                for r in results
                if not r.passed and not r.skipped and r.severity.value == "warning"
            ),
        )
        return results

    def _evaluate_rule(
        self,
        rule: QualityRule,
        report: EvaluationReport,
        signals: dict[str, float | None],
    ) -> QualityRuleResult:
        """Evaluate a single rule."""
        observed = self._resolve_observed(rule, report, signals)
        if observed is None:
            if self._thresholds.skip_missing_metrics and rule.metric_aliases:
                return QualityRuleResult(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    operator=rule.operator,
                    threshold=rule.threshold,
                    observed=None,
                    passed=True,
                    severity=rule.severity,
                    skipped=True,
                    message=f"{rule.name}: skipped (metric not present)",
                    details={"reason": "missing_metric"},
                )
            # Non-metric rules with missing signals: treat as fail for safety
            # when the rule expects a value (e.g. cost when cost is None → skip).
            if rule.rule_id in {
                QualityRuleId.MAX_COST,
                QualityRuleId.MAX_TOOL_LATENCY,
                QualityRuleId.MAX_TOOL_FAILURES,
            }:
                return QualityRuleResult(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    operator=rule.operator,
                    threshold=rule.threshold,
                    observed=None,
                    passed=True,
                    severity=rule.severity,
                    skipped=True,
                    message=f"{rule.name}: skipped (signal not present)",
                    details={"reason": "missing_signal"},
                )

        passed = self._compare(rule.operator, observed, rule.threshold)
        op_label = ">=" if rule.operator == RuleOperator.MINIMUM else "<="
        message = (
            f"{rule.name}: observed={observed} {op_label} {rule.threshold} "
            f"→ {'PASS' if passed else 'FAIL'}"
        )
        return QualityRuleResult(
            rule_id=rule.rule_id,
            name=rule.name,
            operator=rule.operator,
            threshold=rule.threshold,
            observed=observed,
            passed=passed,
            severity=rule.severity,
            skipped=False,
            message=message,
            details={},
        )

    def _resolve_observed(
        self,
        rule: QualityRule,
        report: EvaluationReport,
        signals: dict[str, float | None],
    ) -> float | None:
        """Resolve the observed numeric value for a rule."""
        if rule.rule_id == QualityRuleId.MIN_OVERALL_SCORE:
            return float(report.overall_score)
        if rule.rule_id == QualityRuleId.MAX_TOOL_FAILURES:
            return signals.get("tool_failure_count")
        if rule.rule_id == QualityRuleId.MAX_TOOL_LATENCY:
            return signals.get("max_tool_latency_ms")
        if rule.rule_id == QualityRuleId.MAX_TOTAL_LATENCY:
            return signals.get("total_latency_ms")
        if rule.rule_id == QualityRuleId.MAX_COST:
            return signals.get("cost_usd")

        scores = {m.name: m.score for m in report.metrics}
        for alias in rule.metric_aliases:
            if alias == "overall_score":
                return float(report.overall_score)
            if alias in scores:
                score = float(scores[alias])
                # EvaluationReport stores hallucination on the platform's
                # higher-is-better scale (DeepEval raw rate is inverted in
                # the adapter). MAX_HALLUCINATION thresholds are configured
                # as a raw ceiling (e.g. 0.3), so convert before compare.
                if rule.rule_id == QualityRuleId.MAX_HALLUCINATION:
                    return round(1.0 - score, 6)
                return score
        return None

    @staticmethod
    def _compare(operator: RuleOperator, observed: float | None, threshold: float) -> bool:
        """Compare observed value against threshold."""
        if observed is None:
            return False
        if operator == RuleOperator.MINIMUM:
            return observed >= threshold
        if operator == RuleOperator.MAXIMUM:
            return observed <= threshold
        return False

    @staticmethod
    def _resolve_signals(
        report: EvaluationReport,
        *,
        tool_failure_count: int | None,
        max_tool_latency_ms: float | None,
        total_latency_ms: float | None,
        cost_usd: float | None,
    ) -> dict[str, float | None]:
        """Derive operational signals from the report and overrides."""
        failures = tool_failure_count
        tool_latency = max_tool_latency_ms
        if report.tool_validation is not None:
            summary = report.tool_validation
            if failures is None:
                # Prefer explicit failure lists when available.
                fail_msgs = getattr(summary, "failures", None) or []
                if fail_msgs:
                    failures = len(fail_msgs)
                elif summary.passed is False:
                    failures = 1
                else:
                    failures = 0

        total = total_latency_ms
        if total is None:
            total = (
                report.rag_latency_ms
                if report.rag_latency_ms is not None
                else report.latency
            )

        cost = cost_usd
        if cost is None:
            cost = report.estimated_cost_usd

        # Tool latency from metadata when not supplied.
        if tool_latency is None:
            meta = report.metadata or {}
            raw = meta.get("max_tool_latency_ms")
            if raw is not None:
                try:
                    tool_latency = float(raw)
                except (TypeError, ValueError):
                    tool_latency = None

        return {
            "tool_failure_count": (
                float(failures) if failures is not None else None
            ),
            "max_tool_latency_ms": tool_latency,
            "total_latency_ms": total,
            "cost_usd": float(cost) if cost is not None else None,
        }


__all__ = ["QualityGate"]
