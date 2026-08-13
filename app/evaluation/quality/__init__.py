"""Quality-gate package — Continuous AI Evaluation decision layer.

Sprint 12: QualityGate → QualityGateEngine → QualityDecision (PASS/WARNING/FAIL)
with configurable rules, recommendations, and multi-format reports.
"""

from app.evaluation.quality.decision import (
    QualityDecision,
    QualityRecommendation,
    QualityStatus,
)
from app.evaluation.quality.engine import QualityGateEngine
from app.evaluation.quality.exceptions import (
    BenchmarkHistoryError,
    InvalidQualityDecisionError,
    InvalidQualityRuleError,
    QualityGateDisabledError,
    QualityGateError,
)
from app.evaluation.quality.gate import QualityGate
from app.evaluation.quality.models import (
    QualityGateThresholds,
    QualityRule,
    QualityRuleId,
    QualityRuleResult,
    RuleOperator,
    RuleSeverity,
    build_default_rules,
)
from app.evaluation.quality.report import (
    QualityReportWriter,
    build_quality_report_payload,
)
from app.evaluation.quality.recommendations import recommendations_for_results

__all__ = [
    "QualityStatus",
    "QualityRecommendation",
    "QualityDecision",
    "QualityGate",
    "QualityGateEngine",
    "QualityGateThresholds",
    "QualityRule",
    "QualityRuleId",
    "QualityRuleResult",
    "RuleOperator",
    "RuleSeverity",
    "build_default_rules",
    "recommendations_for_results",
    "QualityReportWriter",
    "build_quality_report_payload",
    "QualityGateError",
    "QualityGateDisabledError",
    "InvalidQualityRuleError",
    "InvalidQualityDecisionError",
    "BenchmarkHistoryError",
]
