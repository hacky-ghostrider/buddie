"""Report writers — JSON, CSV, and HTML evaluation artifacts.

Structured multi-format reports are the evaluation equivalent of JUnit XML +
Allure HTML: machines compare JSON/CSV; humans skim HTML.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable

from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


REPORT_ROW_FIELDS: tuple[str, ...] = (
    "question",
    "expected_answer",
    "actual_answer",
    "deepeval_metrics",
    "langsmith_trace_url",
    "langsmith_run_id",
    "langsmith_trace_id",
    "retrieved_documents",
    "expected_tools",
    "actual_tools",
    "tool_validation_passed",
    "latency_ms",
    "token_usage",
    "estimated_cost_usd",
    "overall_score",
    "passed",
    "timestamp",
)


def report_to_row(report: EvaluationReport) -> dict[str, Any]:
    """Flatten one ``EvaluationReport`` into a tabular / export row."""
    metrics = {m.name: m.score for m in report.metrics}
    tool = report.tool_validation
    return {
        "question": report.question,
        "expected_answer": report.expected_answer,
        "actual_answer": report.answer,
        "deepeval_metrics": json.dumps(metrics),
        "langsmith_trace_url": report.langsmith_run_url,
        "langsmith_run_id": report.langsmith_run_id,
        "langsmith_trace_id": report.langsmith_trace_id,
        "retrieved_documents": json.dumps(
            [
                {"id": d.id, "score": d.score, "text": d.text[:200]}
                for d in report.retrieved_documents
            ]
        ),
        "expected_tools": (
            json.dumps(tool.expected_tools) if tool is not None else "[]"
        ),
        "actual_tools": (
            json.dumps(tool.actual_tools) if tool is not None else "[]"
        ),
        "tool_validation_passed": (
            tool.passed if tool is not None else None
        ),
        "latency_ms": report.rag_latency_ms
        if report.rag_latency_ms is not None
        else report.latency,
        "token_usage": json.dumps(report.token_usage),
        "estimated_cost_usd": report.estimated_cost_usd,
        "overall_score": report.overall_score,
        "passed": report.passed,
        "timestamp": report.evaluation_time.isoformat(),
    }


class EvaluationReportWriter:
    """Write evaluation reports to JSON, CSV, and HTML.

    Args:
        output_directory: Target directory (created if missing).
    """

    def __init__(self, output_directory: str | Path) -> None:
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

    def write_all(
        self,
        reports: Iterable[EvaluationReport],
        *,
        run_name: str | None = None,
    ) -> dict[str, Path]:
        """Write JSON + CSV + HTML for a batch of reports.

        Args:
            reports: Evaluation reports.
            run_name: Optional stem for output files.

        Returns:
            Map of format → written path.
        """
        materialised = list(reports)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = run_name or f"evaluation_{stamp}"
        paths = {
            "json": self.write_json(materialised, stem),
            "csv": self.write_csv(materialised, stem),
            "html": self.write_html(materialised, stem),
        }
        logger.info(
            "Wrote evaluation reports: count=%d paths=%s",
            len(materialised),
            {k: str(v) for k, v in paths.items()},
        )
        return paths

    def write_json(
        self,
        reports: list[EvaluationReport],
        stem: str,
    ) -> Path:
        """Serialize reports to a JSON array file."""
        path = self._output_directory / f"{stem}.json"
        payload = [r.model_dump(mode="json") for r in reports]
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def write_csv(
        self,
        reports: list[EvaluationReport],
        stem: str,
    ) -> Path:
        """Serialize flattened report rows to CSV."""
        path = self._output_directory / f"{stem}.csv"
        rows = [report_to_row(r) for r in reports]
        fieldnames = list(REPORT_ROW_FIELDS)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return path

    def write_html(
        self,
        reports: list[EvaluationReport],
        stem: str,
    ) -> Path:
        """Serialize a simple HTML table report for human review."""
        path = self._output_directory / f"{stem}.html"
        rows_html: list[str] = []
        for report in reports:
            row = report_to_row(report)
            rows_html.append(
                "<tr>"
                f"<td>{escape(str(row['question']))}</td>"
                f"<td>{escape(str(row['expected_answer'] or ''))}</td>"
                f"<td>{escape(str(row['actual_answer']))}</td>"
                f"<td>{escape(str(row['deepeval_metrics']))}</td>"
                f"<td>{escape(str(row['langsmith_trace_url'] or ''))}</td>"
                f"<td>{escape(str(row['retrieved_documents']))}</td>"
                f"<td>{escape(str(row['expected_tools']))}</td>"
                f"<td>{escape(str(row['actual_tools']))}</td>"
                f"<td>{escape(str(row['tool_validation_passed']))}</td>"
                f"<td>{escape(str(row['latency_ms']))}</td>"
                f"<td>{escape(str(row['token_usage']))}</td>"
                f"<td>{escape(str(row['overall_score']))}</td>"
                f"<td>{escape(str(row['passed']))}</td>"
                f"<td>{escape(str(row['timestamp']))}</td>"
                "</tr>"
            )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Evaluation Report — {escape(stem)}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; background: #f7f5f1; color: #1c1b19; }}
    h1 {{ font-size: 1.6rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; background: #fff; }}
    th, td {{ border: 1px solid #d6d0c4; padding: 0.45rem 0.55rem; vertical-align: top; }}
    th {{ background: #ebe4d7; text-align: left; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #faf8f4; }}
  </style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <p>Run: {escape(stem)} — {len(reports)} example(s)</p>
  <table>
    <thead>
      <tr>
        <th>Question</th><th>Expected Answer</th><th>Actual Answer</th>
        <th>DeepEval Metrics</th><th>LangSmith Trace URL</th>
        <th>Retrieved Documents</th><th>Expected Tool</th><th>Actual Tool</th>
        <th>Tool Validation</th><th>Latency</th><th>Token Usage</th>
        <th>Overall Score</th><th>Pass / Fail</th><th>Timestamp</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path


__all__ = ["EvaluationReportWriter", "report_to_row", "REPORT_ROW_FIELDS"]
