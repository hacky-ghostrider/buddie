"""Benchmark aggregation — suite-level averages for quality and cost.

WHY
---
Per-question reports are too noisy for leadership / release gates.
Benchmarks roll up faithfulness, hallucination, relevancy, context
precision/recall, latency, tokens, cost, and pass rate into one scorecard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark metrics across an evaluation suite."""

    model_config = ConfigDict(extra="forbid")

    example_count: int = Field(ge=0)
    average_faithfulness: float | None = None
    average_hallucination: float | None = None
    average_relevancy: float | None = None
    average_context_precision: float | None = None
    average_context_recall: float | None = None
    average_latency_ms: float | None = None
    average_tokens: float | None = None
    average_cost_usd: float | None = None
    pass_rate: float = Field(ge=0.0, le=1.0)
    overall_average_score: float | None = None
    metric_averages: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkRunner:
    """Compute benchmark summaries from evaluation reports."""

    _METRIC_ALIASES: dict[str, tuple[str, ...]] = {
        "average_faithfulness": ("faithfulness",),
        "average_hallucination": ("hallucination",),
        "average_relevancy": ("answer_relevancy", "relevancy"),
        "average_context_precision": ("contextual_precision", "context_precision"),
        "average_context_recall": ("contextual_recall", "context_recall"),
    }

    def summarize(
        self,
        reports: Iterable[EvaluationReport],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkSummary:
        """Aggregate reports into a ``BenchmarkSummary``."""
        materialised = list(reports)
        if not materialised:
            return BenchmarkSummary(
                example_count=0,
                pass_rate=0.0,
                metadata=dict(metadata or {}),
            )

        metric_sums: dict[str, list[float]] = {}
        latencies: list[float] = []
        tokens: list[float] = []
        costs: list[float] = []
        overall_scores: list[float] = []
        passed = 0

        for report in materialised:
            overall_scores.append(report.overall_score)
            if report.passed:
                passed += 1
            latency = (
                report.rag_latency_ms
                if report.rag_latency_ms is not None
                else report.latency
            )
            latencies.append(latency)
            total_tokens = report.token_usage.get("total_tokens")
            if total_tokens is not None:
                tokens.append(float(total_tokens))
            if report.estimated_cost_usd is not None:
                costs.append(float(report.estimated_cost_usd))
            for metric in report.metrics:
                metric_sums.setdefault(metric.name, []).append(metric.score)

        metric_averages = {
            name: round(sum(values) / len(values), 6)
            for name, values in metric_sums.items()
            if values
        }

        def _avg_for(field_name: str) -> float | None:
            for alias in self._METRIC_ALIASES[field_name]:
                if alias in metric_averages:
                    return metric_averages[alias]
            return None

        summary = BenchmarkSummary(
            example_count=len(materialised),
            average_faithfulness=_avg_for("average_faithfulness"),
            average_hallucination=_avg_for("average_hallucination"),
            average_relevancy=_avg_for("average_relevancy"),
            average_context_precision=_avg_for("average_context_precision"),
            average_context_recall=_avg_for("average_context_recall"),
            average_latency_ms=round(sum(latencies) / len(latencies), 3),
            average_tokens=(
                round(sum(tokens) / len(tokens), 3) if tokens else None
            ),
            average_cost_usd=(
                round(sum(costs) / len(costs), 8) if costs else None
            ),
            pass_rate=round(passed / len(materialised), 6),
            overall_average_score=round(
                sum(overall_scores) / len(overall_scores),
                6,
            ),
            metric_averages=metric_averages,
            metadata=dict(metadata or {}),
        )
        logger.info(
            "Benchmark summary: examples=%d pass_rate=%.4f overall=%.4f",
            summary.example_count,
            summary.pass_rate,
            summary.overall_average_score or 0.0,
        )
        return summary

    def write_summary(
        self,
        summary: BenchmarkSummary,
        output_directory: str | Path,
        *,
        run_name: str = "benchmark",
    ) -> Path:
        """Persist a benchmark summary JSON file."""
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{run_name}.json"
        path.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote benchmark summary: path=%s", path)
        return path


__all__ = ["BenchmarkSummary", "BenchmarkRunner"]
