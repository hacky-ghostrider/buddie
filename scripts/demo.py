#!/usr/bin/env python3
"""One-command interview demonstration — Sprint 13 primary demo.

Workflow (offline by default — no API keys required):

    Load agent-tools-foundation-001
            ↓
    Run LangGraph Agent (Planner → Router → Tools)
            ↓
    Generate LangSmith Trace (NoOp or live)
            ↓
    Run DeepEval adapters (deterministic offline scores)
            ↓
    Run Tool Validation
            ↓
    Run Quality Gates → PASS / WARNING / FAIL
            ↓
    Generate Reports + Benchmark
            ↓
    Display Summary

Usage:
    python scripts/demo.py
    python scripts/demo.py --live          # real RAG / DeepEval / LangSmith
    make demo
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
from app.demo.runner import run_canonical_demo
from app.evaluation.deepeval import DeepEvalMetricName
from app.evaluation.quality.decision import QualityStatus
from app.evaluation.scenarios import CANONICAL_DATASET_PATH, CANONICAL_SCENARIO_ID

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interview demo: Agent → LangSmith → DeepEval → Gates → Reports",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real RAG / DeepEval / LangSmith (requires API keys)",
    )
    parser.add_argument(
        "--dataset",
        default=str(CANONICAL_DATASET_PATH),
        help="Path to agent-tools-foundation dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/demo",
        help="Directory for demo evaluation / quality / benchmark artifacts",
    )
    return parser.parse_args(argv)


def _print_banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n  {title}\n{line}")


def _print_summary(result) -> None:
    decision = result.quality_decision
    report = result.evaluation_report
    agent_result = result.agent_result
    tools = [e.tool_name for e in agent_result.tool_executions]
    _print_banner("DEMO SUMMARY — RAG Evaluation Platform v1.0")
    print(f"  Scenario:          {result.scenario_id}")
    print(f"  Mode:              {result.mode}")
    print(f"  Question:          {result.question}")
    print(f"  Answer preview:    {agent_result.final_answer[:100]}...")
    print()
    print("  LangGraph Agent")
    print(f"    Planner tools:   {tools}")
    print(f"    Correlation id:  {agent_result.correlation_id}")
    print()
    print("  LangSmith Trace")
    print(f"    Enabled:         {agent_result.metadata.get('trace_enabled', False)}")
    print(f"    Run id:          {agent_result.run_id or '(noop)'}")
    print(f"    URL:             {agent_result.run_url or '(set ENABLE_LANGSMITH=true for live)'}")
    print()
    print("  DeepEval Metrics")
    for m in report.metrics:
        if m.name in {n.value for n in DeepEvalMetricName} or m.name in {
            "faithfulness",
            "hallucination",
            "answer_relevancy",
            "contextual_precision",
            "contextual_recall",
        }:
            status = "PASS" if m.passed else "FAIL"
            print(f"    {m.name:24s}  {m.score:.3f}  [{status}]")
    print(f"    Overall score:   {report.overall_score:.3f}")
    print()
    print("  Tool Validation")
    if report.tool_validation:
        tv = report.tool_validation
        print(f"    Passed:          {tv.passed}")
        print(f"    Expected:        {tv.expected_tools}")
        print(f"    Actual:          {tv.actual_tools}")
    else:
        print("    (no tool validation report)")
    print()
    print("  Quality Gates")
    print(f"    Decision:        {decision.status.value}")
    print(f"    Reason:          {decision.reason}")
    print(f"    Failed rules:    {decision.failed_rules or '[]'}")
    print(f"    Warnings:        {decision.warnings or '[]'}")
    if decision.recommendations:
        print("    Recommendations:")
        for rec in decision.recommendations[:3]:
            print(f"      - {rec.message}")
    print()
    print("  Artifacts")
    for key, path in sorted(result.report_paths.items()):
        print(f"    {key:16s}  {path}")
    print()
    if decision.status == QualityStatus.FAIL:
        print("  Result: FAIL — quality gates blocked this run.")
    elif decision.status == QualityStatus.WARNING:
        print("  Result: WARNING — review recommendations before release.")
    else:
        print("  Result: PASS — demo completed successfully.")
    print()


def run_demo(*, live: bool, dataset: str, output_dir: str) -> int:
    """Execute the full interview demonstration pipeline (CLI wrapper)."""
    setup_logging(get_settings().log_level)
    _print_banner("STEP 1–6 — Canonical demo pipeline")
    result = run_canonical_demo(live=live, dataset=dataset, output_dir=output_dir)
    print(f"  Scenario: {result.scenario_id}")
    print(f"  Expected tools: {result.expected_tools}")
    print(f"  Actual tools: {[e.tool_name for e in result.agent_result.tool_executions]}")
    print(f"  Quality: {result.quality_decision.status.value}")
    for key, path in sorted(result.report_paths.items()):
        print(f"  {key}: {path}")
    _print_summary(result)
    if result.quality_decision.status == QualityStatus.FAIL:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    try:
        return run_demo(
            live=args.live,
            dataset=args.dataset,
            output_dir=args.output_dir,
        )
    except Exception:  # noqa: BLE001 — demo should surface a clean failure
        logger.exception("Demo failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
