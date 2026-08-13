"""Quality report writers — JSON, CSV, and HTML gate artifacts.

Includes quality decision, benchmark history, regression summary,
DeepEval metrics, tool validation, LangSmith URL, and recommendations.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from app.evaluation.quality.decision import QualityDecision
from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


QUALITY_CSV_FIELDS: tuple[str, ...] = (
    "status",
    "reason",
    "overall_score",
    "failed_rules",
    "warnings",
    "correlation_id",
    "timestamp",
    "question",
    "langsmith_url",
    "tool_validation_passed",
    "recommendation_count",
    "has_regressions",
)


class QualityReportBundle(dict[str, Any]):
    """Typed-ish dict payload for quality report serialization."""


def build_quality_report_payload(
    *,
    decision: QualityDecision,
    reports: list[EvaluationReport] | None = None,
    benchmark_history: dict[str, Any] | None = None,
    regression_summary: dict[str, Any] | None = None,
    recommendations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical quality report document.

    Args:
        decision: Quality gate decision.
        reports: Optional underlying evaluation reports.
        benchmark_history: Optional history snapshot / comparison.
        regression_summary: Optional regression engine summary.
        recommendations: Optional override; defaults to decision recommendations.

    Returns:
        JSON-serializable quality report dict.
    """
    materialised = list(reports or [])
    first = materialised[0] if materialised else None
    recs = recommendations
    if recs is None:
        recs = [r.model_dump(mode="json") for r in decision.recommendations]

    deepeval_metrics: list[dict[str, Any]] = []
    tool_validation: dict[str, Any] | None = None
    langsmith_url: str | None = None
    if first is not None:
        deepeval_metrics = [
            {"name": m.name, "score": m.score, "passed": m.passed}
            for m in first.metrics
        ]
        if first.tool_validation is not None:
            tool_validation = first.tool_validation.to_summary_dict()
        langsmith_url = first.langsmith_run_url

    return {
        "quality_decision": decision.model_dump(mode="json"),
        "status": decision.status.value,
        "reason": decision.reason,
        "overall_score": decision.overall_score,
        "failed_rules": list(decision.failed_rules),
        "warnings": list(decision.warnings),
        "correlation_id": decision.correlation_id,
        "timestamp": decision.timestamp.isoformat(),
        "deepeval_metrics": deepeval_metrics,
        "tool_validation": tool_validation,
        "langsmith_url": langsmith_url
        or (decision.metadata or {}).get("langsmith_run_url"),
        "benchmark_history": benchmark_history or {},
        "regression_summary": regression_summary or {},
        "recommendations": recs,
        "evaluation_reports": [
            {
                "question": r.question,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "metrics": {m.name: m.score for m in r.metrics},
                "langsmith_run_url": r.langsmith_run_url,
                "estimated_cost_usd": r.estimated_cost_usd,
                "latency_ms": r.rag_latency_ms
                if r.rag_latency_ms is not None
                else r.latency,
            }
            for r in materialised
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


class QualityReportWriter:
    """Persist quality_report.json / .csv / .html.

    Args:
        output_directory: Target directory (created if missing).
    """

    def __init__(self, output_directory: str | Path) -> None:
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        *,
        decision: QualityDecision,
        reports: list[EvaluationReport] | None = None,
        benchmark_history: dict[str, Any] | None = None,
        regression_summary: dict[str, Any] | None = None,
        run_name: str = "quality_report",
    ) -> dict[str, Path]:
        """Write JSON, CSV, and HTML quality reports.

        Args:
            decision: Quality decision.
            reports: Optional evaluation reports.
            benchmark_history: Optional history payload.
            regression_summary: Optional regression payload.
            run_name: Output file stem (default ``quality_report``).

        Returns:
            Map of format → path.
        """
        payload = build_quality_report_payload(
            decision=decision,
            reports=reports,
            benchmark_history=benchmark_history,
            regression_summary=regression_summary,
        )
        paths = {
            "json": self.write_json(payload, run_name=run_name),
            "csv": self.write_csv(payload, run_name=run_name),
            "html": self.write_html(payload, run_name=run_name),
        }
        logger.info(
            "Wrote quality reports: status=%s paths=%s",
            decision.status.value,
            {k: str(v) for k, v in paths.items()},
        )
        return paths

    def write_json(self, payload: dict[str, Any], *, run_name: str) -> Path:
        """Write quality_report.json."""
        path = self._output_directory / f"{run_name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_csv(self, payload: dict[str, Any], *, run_name: str) -> Path:
        """Write quality_report.csv (one summary row + recommendation rows)."""
        path = self._output_directory / f"{run_name}.csv"
        row = {
            "status": payload.get("status"),
            "reason": payload.get("reason"),
            "overall_score": payload.get("overall_score"),
            "failed_rules": json.dumps(payload.get("failed_rules") or []),
            "warnings": json.dumps(payload.get("warnings") or []),
            "correlation_id": payload.get("correlation_id"),
            "timestamp": payload.get("timestamp"),
            "question": (
                (payload.get("evaluation_reports") or [{}])[0].get("question")
                if payload.get("evaluation_reports")
                else None
            ),
            "langsmith_url": payload.get("langsmith_url"),
            "tool_validation_passed": (
                (payload.get("tool_validation") or {}).get("passed")
                if payload.get("tool_validation")
                else None
            ),
            "recommendation_count": len(payload.get("recommendations") or []),
            "has_regressions": bool(
                (payload.get("regression_summary") or {}).get("has_regressions")
            ),
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QUALITY_CSV_FIELDS))
            writer.writeheader()
            writer.writerow(row)
        return path

    def write_html(self, payload: dict[str, Any], *, run_name: str) -> Path:
        """Write quality_report.html for human review."""
        path = self._output_directory / f"{run_name}.html"
        status = str(payload.get("status") or "UNKNOWN")
        color = {"PASS": "#1b7f4e", "WARNING": "#b36b00", "FAIL": "#b00020"}.get(
            status, "#333"
        )
        recommendations = payload.get("recommendations") or []
        rec_items = "".join(
            f"<li><strong>{escape(str(r.get('category')))}</strong>: "
            f"{escape(str(r.get('message')))}"
            f"<ul>{''.join(f'<li>{escape(a)}</li>' for a in (r.get('actions') or []))}</ul>"
            f"</li>"
            for r in recommendations
        ) or "<li>None</li>"

        metrics = payload.get("deepeval_metrics") or []
        metric_rows = "".join(
            f"<tr><td>{escape(str(m.get('name')))}</td>"
            f"<td>{escape(str(m.get('score')))}</td>"
            f"<td>{escape(str(m.get('passed')))}</td></tr>"
            for m in metrics
        ) or "<tr><td colspan='3'>No metrics</td></tr>"

        tool_json = json.dumps(payload.get("tool_validation") or {}, indent=2)
        regression_json = json.dumps(
            payload.get("regression_summary") or {},
            indent=2,
        )
        history_json = json.dumps(payload.get("benchmark_history") or {}, indent=2)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Quality Report — {escape(status)}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #222; }}
    h1 {{ color: {color}; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .meta {{ color: #555; }}
    code {{ background: #f0f0f0; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>Quality Decision: {escape(status)}</h1>
  <p class="meta">{escape(str(payload.get("reason")))}</p>
  <p class="meta">
    Score: <code>{escape(str(payload.get("overall_score")))}</code> ·
    Correlation: <code>{escape(str(payload.get("correlation_id")))}</code> ·
    Timestamp: <code>{escape(str(payload.get("timestamp")))}</code>
  </p>
  <p class="meta">
    LangSmith:
    <a href="{escape(str(payload.get("langsmith_url") or "#"))}">
      {escape(str(payload.get("langsmith_url") or "n/a"))}
    </a>
  </p>

  <h2>Failed Rules</h2>
  <p>{escape(json.dumps(payload.get("failed_rules") or []))}</p>
  <h2>Warnings</h2>
  <p>{escape(json.dumps(payload.get("warnings") or []))}</p>

  <h2>DeepEval Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Score</th><th>Passed</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>

  <h2>Tool Validation</h2>
  <pre>{escape(tool_json)}</pre>

  <h2>Regression Summary</h2>
  <pre>{escape(regression_json)}</pre>

  <h2>Benchmark History</h2>
  <pre>{escape(history_json)}</pre>

  <h2>Recommendations</h2>
  <ul>{rec_items}</ul>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path


__all__ = [
    "QUALITY_CSV_FIELDS",
    "build_quality_report_payload",
    "QualityReportWriter",
]
