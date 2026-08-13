#!/usr/bin/env python3
"""CLI: compare previous vs current evaluation runs for regressions.

Highlights:
    - Score regressions
    - Latency regressions
    - Tool regressions
    - Prompt / answer regressions
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
from app.evaluation.regression import RegressionRunner

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation regression comparison")
    parser.add_argument(
        "--previous",
        required=True,
        help="Path to previous evaluation JSON report array",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to current evaluation JSON report array",
    )
    parser.add_argument(
        "--score-drop",
        type=float,
        default=0.05,
        help="Absolute score drop that counts as a regression",
    )
    parser.add_argument(
        "--latency-increase-ratio",
        type=float,
        default=0.25,
        help="Relative latency increase that counts as a regression",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write RegressionReport JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for regression comparison."""
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    runner = RegressionRunner(
        score_drop_threshold=args.score_drop,
        latency_increase_ratio=args.latency_increase_ratio,
    )
    report = runner.compare_files(args.previous, args.current)
    payload = report.model_dump(mode="json")
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote regression report: path=%s", out)
    logger.info(
        "Regression result: has_regressions=%s summary=%s",
        report.has_regressions,
        report.summary,
    )
    return 1 if report.has_regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
