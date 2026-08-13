#!/usr/bin/env python3
"""CLI: aggregate evaluation reports into a benchmark scorecard.

Outputs averages for faithfulness, hallucination, relevancy, context
precision/recall, latency, tokens, cost, and pass rate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation benchmarking")
    parser.add_argument(
        "--reports",
        required=True,
        help="Path to evaluation JSON report array",
    )
    parser.add_argument(
        "--run-name",
        default="benchmark",
        help="Output benchmark file stem",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Benchmark output directory (default from settings)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for benchmarking."""
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    path = Path(args.reports)
    raw = json.loads(path.read_text(encoding="utf-8"))
    reports = [EvaluationReport.model_validate(item) for item in raw]

    runner = BenchmarkRunner()
    summary = runner.summarize(
        reports,
        metadata={"source_reports": str(path)},
    )
    out_dir = args.output_dir or settings.benchmark_directory
    out_path = runner.write_summary(summary, out_dir, run_name=args.run_name)
    logger.info(
        "Benchmark complete: path=%s pass_rate=%.4f",
        out_path,
        summary.pass_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
