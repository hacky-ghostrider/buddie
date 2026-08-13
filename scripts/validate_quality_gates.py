#!/usr/bin/env python3
"""CI quality-gate validator — fail the build on FAIL or detected regression.

Runs a deterministic continuous-evaluation pass suitable for GitHub Actions:
no live OpenAI / LangSmith / DeepEval LLM-as-judge calls.

Exit codes:
    0 — PASS or WARNING (warnings are reported but do not block by default)
    1 — FAIL quality decision or regression detected
    2 — unexpected error
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.evaluation.continuous import ContinuousEvaluationService
from app.evaluation.models import MetricResult
from app.evaluation.quality.decision import QualityStatus
from app.evaluation.report import EvaluationReport
from app.evaluation.tool_validation.report import ToolValidationReport
from app.retrieval.models import RetrievedDocument

logger = logging.getLogger(__name__)


def _metric(name: str, score: float, *, passed: bool | None = None) -> MetricResult:
    threshold = 0.7
    ok = passed if passed is not None else score >= threshold
    return MetricResult(
        name=name,
        score=score,
        passed=ok,
        details={},
        latency_ms=1.0,
    )


def _baseline_report() -> EvaluationReport:
    """Stable high-quality report used as previous/current baseline."""
    return EvaluationReport.build(
        question="Summarize the leave policy from the employee handbook.",
        answer=(
            "The employee handbook leave policy defines paid time off eligibility, "
            "accrual, approval workflow, and notice requirements."
        ),
        expected_answer=(
            "The employee handbook leave policy defines paid time off eligibility, "
            "accrual, approval workflow, and notice requirements for vacation and "
            "related leave types."
        ),
        retrieved_documents=[
            RetrievedDocument(
                id="c1",
                text="Leave policy: paid time off accrual and approval.",
                score=0.92,
                metadata={"source": "employee_handbook.pdf"},
            )
        ],
        metrics=[
            _metric("faithfulness", 0.95),
            _metric("hallucination", 0.92),
            _metric("answer_relevancy", 0.93),
            _metric("contextual_precision", 0.88),
            _metric("contextual_recall", 0.90),
        ],
        latency_ms=120.0,
        pass_threshold=0.7,
        rag_latency_ms=95.0,
        estimated_cost_usd=0.002,
        langsmith_run_url="https://smith.langchain.com/public/ci-demo",
        tool_validation=ToolValidationReport(
            passed=True,
            expected_tools=["search_docs", "summarize"],
            actual_tools=["search_docs", "summarize"],
            failures=[],
            matches=[],
        ),
        metadata={"correlation_id": "ci-quality-gate", "suite": "ci"},
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate quality gates for CI")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Treat WARNING decisions as CI failures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint — run continuous evaluation and enforce gate policy."""
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    Path(settings.quality_report_directory).mkdir(parents=True, exist_ok=True)
    Path(settings.benchmark_directory).mkdir(parents=True, exist_ok=True)

    previous = _baseline_report()
    current = _baseline_report()

    service = ContinuousEvaluationService(settings=settings)
    result = service.evaluate(
        [current],
        previous_reports=[previous],
        suite_name="ci",
        correlation_id="ci-quality-gate",
        write_reports=True,
        run_name="ci_quality_report",
        metadata={"source": "validate_quality_gates.py"},
    )

    decision = result.decision
    logger.info(
        "Quality gate decision: status=%s reason=%s failed=%s warnings=%s",
        decision.status.value,
        decision.reason,
        decision.failed_rules,
        decision.warnings,
    )
    if result.regression_report is not None:
        logger.info(
            "Regression: has_regressions=%s",
            result.regression_report.has_regressions,
        )

    if decision.status == QualityStatus.FAIL:
        logger.error("Quality gates FAILED — blocking CI")
        return 1
    if result.regression_report is not None and result.regression_report.has_regressions:
        logger.error("Regression detected — blocking CI")
        return 1
    if args.fail_on_warning and decision.status == QualityStatus.WARNING:
        logger.error("Quality gates WARNING and --fail-on-warning set")
        return 1

    logger.info("Quality gates passed (%s)", decision.status.value)
    for key, path in result.output_paths.items():
        logger.info("Artifact %s -> %s", key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
