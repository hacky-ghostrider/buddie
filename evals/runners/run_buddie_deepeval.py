"""CLI: run Buddie evaluation suite on the frozen golden dataset.

Uses ``app.api.deps.get_agent_service`` and Gemini (DeepEval ``GeminiModel``)
when ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` is set. Does not hard-code or
print API keys.

Example:
    python -m evals.runners.run_buddie_deepeval
    python -m evals.runners.run_buddie_deepeval --output data/reports/buddie_eval_suite.json
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Buddie AI evaluation suite (golden cases)",
    )
    parser.add_argument(
        "--output",
        default="data/reports/buddie_eval_suite.json",
        help="JSON report output path",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        default=None,
        help="Optional case id filter (repeatable)",
    )
    parser.add_argument(
        "--tier",
        choices=("smoke", "sanity", "regression"),
        default=None,
        help="Run only golden cases tagged for this CI tier",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.api.deps import get_agent_service
    from evals.metrics.annotations import (
        build_annotation_report,
        format_annotation_console,
    )
    from evals.metrics.config import (
        default_buddie_deepeval_config,
        ensure_gemini_env_loaded,
        gemini_judge_status,
        require_deepeval_judge_model,
    )
    from evals.runners.deepeval_suite import (
        format_suite_console,
        run_buddie_eval_suite,
        write_suite_report_json,
    )

    # Load .env + normalize GOOGLE_API_KEY / GEMINI_API_KEY before judge resolve.
    ensure_gemini_env_loaded()
    status = gemini_judge_status()
    print(f"Gemini configured: {status['configured']}", flush=True)
    print(f"Gemini model: {status['model_name'] or 'N/A'}", flush=True)
    print(f"LLM judge provider: {status['provider']}", flush=True)
    if not status["configured"]:
        print(
            "ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY so DeepEval metrics "
            "use GeminiModel (otherwise they fall back to GPTModel).",
            flush=True,
        )
        return 1

    # Fail fast if GeminiModel cannot be constructed (never print the key).
    require_deepeval_judge_model()
    config = default_buddie_deepeval_config()
    if config.model is None:
        print(
            "ERROR: BuddieDeepEvalConfig.model is None after Gemini resolve.",
            flush=True,
        )
        return 1

    agent = get_agent_service()
    report = run_buddie_eval_suite(
        agent,
        config=config,
        case_ids=args.case_ids,
        test_tier=args.tier,
    )
    write_suite_report_json(report, args.output)
    print(format_annotation_console(build_annotation_report()))
    print()
    print(format_suite_console(report))
    print(f"\nWrote JSON report: {args.output}")
    if report.failed or report.errors or report.rate_limited:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
