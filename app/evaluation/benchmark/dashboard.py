"""Benchmark dashboard models — data contracts for future UIs.

Sprint 12 stores historical scores, trends, averages, and failure counts.
A web dashboard is intentionally deferred; these models are the API surface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.benchmark.history import BenchmarkRunRecord


class TrendPoint(BaseModel):
    """One point on a historical trend series."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: datetime
    suite_name: str
    average_score: float | None = None
    pass_rate: float = 0.0
    average_latency_ms: float | None = None
    average_cost_usd: float | None = None
    quality_status: str | None = None


class BenchmarkDashboardModel(BaseModel):
    """Aggregated view supporting future dashboards (no web implementation).

    Attributes:
        evaluation_count: Total historical runs.
        failure_count: Runs with FAIL quality status.
        warning_count: Runs with WARNING quality status.
        pass_count: Runs with PASS quality status.
        average_metrics: Mean of key metrics across history.
        historical_scores: Series of overall scores.
        trend_data: Richer trend points for charts.
        latest_run_id: Newest run id when present.
        metadata: Free-form extras.
    """

    model_config = ConfigDict(extra="forbid")

    evaluation_count: int = Field(ge=0, default=0)
    failure_count: int = Field(ge=0, default=0)
    warning_count: int = Field(ge=0, default=0)
    pass_count: int = Field(ge=0, default=0)
    average_metrics: dict[str, float] = Field(default_factory=dict)
    historical_scores: list[float] = Field(default_factory=list)
    trend_data: list[TrendPoint] = Field(default_factory=list)
    latest_run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_history(
        cls,
        records: list[BenchmarkRunRecord],
    ) -> BenchmarkDashboardModel:
        """Build a dashboard model from history records."""
        if not records:
            return cls()

        failure_count = 0
        warning_count = 0
        pass_count = 0
        scores: list[float] = []
        latencies: list[float] = []
        costs: list[float] = []
        pass_rates: list[float] = []
        trends: list[TrendPoint] = []

        for record in records:
            status = (
                None
                if record.quality_status is None
                else record.quality_status.value
            )
            if status == "FAIL":
                failure_count += 1
            elif status == "WARNING":
                warning_count += 1
            elif status == "PASS":
                pass_count += 1

            if record.average_score is not None:
                scores.append(float(record.average_score))
            if record.average_latency_ms is not None:
                latencies.append(float(record.average_latency_ms))
            if record.average_cost_usd is not None:
                costs.append(float(record.average_cost_usd))
            pass_rates.append(float(record.pass_rate))

            trends.append(
                TrendPoint(
                    run_id=record.run_id,
                    timestamp=record.timestamp,
                    suite_name=record.suite_name,
                    average_score=record.average_score,
                    pass_rate=record.pass_rate,
                    average_latency_ms=record.average_latency_ms,
                    average_cost_usd=record.average_cost_usd,
                    quality_status=status,
                )
            )

        averages: dict[str, float] = {}
        if scores:
            averages["average_score"] = round(sum(scores) / len(scores), 6)
        if latencies:
            averages["average_latency_ms"] = round(
                sum(latencies) / len(latencies),
                3,
            )
        if costs:
            averages["average_cost_usd"] = round(sum(costs) / len(costs), 8)
        if pass_rates:
            averages["average_pass_rate"] = round(
                sum(pass_rates) / len(pass_rates),
                6,
            )

        return cls(
            evaluation_count=len(records),
            failure_count=failure_count,
            warning_count=warning_count,
            pass_count=pass_count,
            average_metrics=averages,
            historical_scores=scores,
            trend_data=trends,
            latest_run_id=records[-1].run_id,
            metadata={"suite_names": sorted({r.suite_name for r in records})},
        )


__all__ = ["TrendPoint", "BenchmarkDashboardModel"]
