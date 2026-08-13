"""Benchmark history — durable store of suite runs for trend / regression.

WHY
---
A single ``BenchmarkSummary`` is a snapshot. Production quality engineering
needs *history*: pass rate over time, score trends, cost/latency drift, and
comparison between the latest run and the previous baseline.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.benchmark.runner import BenchmarkSummary
from app.evaluation.quality.decision import QualityStatus
from app.evaluation.quality.exceptions import BenchmarkHistoryError

logger = logging.getLogger(__name__)


class BenchmarkRunRecord(BaseModel):
    """One historical benchmark / continuous-evaluation run.

    Attributes:
        run_id: Stable run identifier.
        timestamp: UTC time the run was recorded.
        suite_name: Logical suite / scenario name.
        summary: Aggregated ``BenchmarkSummary`` payload.
        quality_status: Optional PASS / WARNING / FAIL.
        pass_rate: Convenience copy of summary pass rate.
        average_score: Convenience copy of overall average.
        average_latency_ms: Convenience latency.
        average_cost_usd: Convenience cost.
        tool_validation_pass_rate: Optional tool-validation pass rate.
        correlation_id: Optional correlation id.
        metadata: Free-form extras.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="Unique run id")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    suite_name: str = Field(default="default", description="Suite name")
    summary: BenchmarkSummary
    quality_status: QualityStatus | None = None
    pass_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    average_score: float | None = None
    average_latency_ms: float | None = None
    average_cost_usd: float | None = None
    tool_validation_pass_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    correlation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "suite_name")
    @classmethod
    def ids_must_not_be_blank(cls, value: str) -> str:
        """Reject blank identifiers."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("run_id / suite_name must be non-empty")
        return cleaned

    @classmethod
    def from_summary(
        cls,
        summary: BenchmarkSummary,
        *,
        suite_name: str = "default",
        run_id: str | None = None,
        quality_status: QualityStatus | None = None,
        tool_validation_pass_rate: float | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkRunRecord:
        """Build a history record from a ``BenchmarkSummary``."""
        return cls(
            run_id=run_id or str(uuid.uuid4()),
            suite_name=suite_name,
            summary=summary,
            quality_status=quality_status,
            pass_rate=summary.pass_rate,
            average_score=summary.overall_average_score,
            average_latency_ms=summary.average_latency_ms,
            average_cost_usd=summary.average_cost_usd,
            tool_validation_pass_rate=tool_validation_pass_rate,
            correlation_id=correlation_id,
            metadata=dict(metadata or {}),
        )


class BenchmarkComparison(BaseModel):
    """Delta between two benchmark runs."""

    model_config = ConfigDict(extra="forbid")

    previous_run_id: str
    current_run_id: str
    score_delta: float | None = None
    latency_delta_ms: float | None = None
    cost_delta_usd: float | None = None
    pass_rate_delta: float | None = None
    tool_validation_delta: float | None = None
    trend: str = Field(
        default="stable",
        description="improving | degrading | stable | unknown",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class BenchmarkHistory:
    """Append-only JSON history store for benchmark runs.

    Args:
        storage_path: File path for the history JSON document.
    """

    def __init__(self, storage_path: str | Path) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Return the storage path."""
        return self._path

    def load(self) -> list[BenchmarkRunRecord]:
        """Load all historical runs (oldest → newest)."""
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BenchmarkHistoryError(
                f"Corrupt benchmark history at {self._path}: {exc}"
            ) from exc
        if not isinstance(raw, list):
            raise BenchmarkHistoryError(
                "Benchmark history must be a JSON array of run records"
            )
        return [BenchmarkRunRecord.model_validate(item) for item in raw]

    def append(self, record: BenchmarkRunRecord) -> BenchmarkRunRecord:
        """Append a run and persist the history file.

        Args:
            record: Run to store.

        Returns:
            The stored record.
        """
        history = self.load()
        history.append(record)
        self._write(history)
        logger.info(
            "Benchmark update: run_id=%s suite=%s status=%s pass_rate=%.4f "
            "score=%s path=%s",
            record.run_id,
            record.suite_name,
            None if record.quality_status is None else record.quality_status.value,
            record.pass_rate,
            record.average_score,
            self._path,
        )
        return record

    def latest(self, *, suite_name: str | None = None) -> BenchmarkRunRecord | None:
        """Return the newest run, optionally filtered by suite."""
        history = self.load()
        if suite_name is not None:
            history = [r for r in history if r.suite_name == suite_name]
        return history[-1] if history else None

    def previous(
        self,
        *,
        suite_name: str | None = None,
    ) -> BenchmarkRunRecord | None:
        """Return the second-newest run (baseline for comparison)."""
        history = self.load()
        if suite_name is not None:
            history = [r for r in history if r.suite_name == suite_name]
        if len(history) < 2:
            return None
        return history[-2]

    def compare(
        self,
        current: BenchmarkRunRecord,
        previous: BenchmarkRunRecord | None = None,
        *,
        suite_name: str | None = None,
        score_drop_threshold: float = 0.05,
        latency_increase_ratio: float = 0.25,
    ) -> BenchmarkComparison | None:
        """Compare ``current`` against ``previous`` (or auto-loaded baseline).

        Returns:
            ``BenchmarkComparison`` or ``None`` when no baseline exists.
        """
        baseline = previous
        if baseline is None:
            baseline = self.previous(suite_name=suite_name or current.suite_name)
        if baseline is None:
            return None

        score_delta = None
        if current.average_score is not None and baseline.average_score is not None:
            score_delta = round(current.average_score - baseline.average_score, 6)

        latency_delta = None
        if (
            current.average_latency_ms is not None
            and baseline.average_latency_ms is not None
        ):
            latency_delta = round(
                current.average_latency_ms - baseline.average_latency_ms,
                3,
            )

        cost_delta = None
        if (
            current.average_cost_usd is not None
            and baseline.average_cost_usd is not None
        ):
            cost_delta = round(
                current.average_cost_usd - baseline.average_cost_usd,
                8,
            )

        pass_delta = round(current.pass_rate - baseline.pass_rate, 6)

        tool_delta = None
        if (
            current.tool_validation_pass_rate is not None
            and baseline.tool_validation_pass_rate is not None
        ):
            tool_delta = round(
                current.tool_validation_pass_rate - baseline.tool_validation_pass_rate,
                6,
            )

        degrading = False
        if score_delta is not None and score_delta <= -score_drop_threshold:
            degrading = True
        if (
            latency_delta is not None
            and baseline.average_latency_ms
            and baseline.average_latency_ms > 0
            and (current.average_latency_ms or 0) / baseline.average_latency_ms - 1.0
            >= latency_increase_ratio
        ):
            degrading = True
        if pass_delta < 0:
            degrading = True

        improving = (
            not degrading
            and (
                (score_delta is not None and score_delta > 0)
                or pass_delta > 0
            )
        )
        if degrading:
            trend = "degrading"
        elif improving:
            trend = "improving"
        else:
            trend = "stable"

        comparison = BenchmarkComparison(
            previous_run_id=baseline.run_id,
            current_run_id=current.run_id,
            score_delta=score_delta,
            latency_delta_ms=latency_delta,
            cost_delta_usd=cost_delta,
            pass_rate_delta=pass_delta,
            tool_validation_delta=tool_delta,
            trend=trend,
            details={
                "previous_pass_rate": baseline.pass_rate,
                "current_pass_rate": current.pass_rate,
                "previous_score": baseline.average_score,
                "current_score": current.average_score,
            },
        )
        logger.info(
            "Benchmark comparison: previous=%s current=%s trend=%s "
            "score_delta=%s pass_rate_delta=%s",
            comparison.previous_run_id,
            comparison.current_run_id,
            comparison.trend,
            comparison.score_delta,
            comparison.pass_rate_delta,
        )
        return comparison

    def to_dashboard_snapshot(self) -> dict[str, Any]:
        """Serialize history into a dashboard-friendly dict (no web UI)."""
        from app.evaluation.benchmark.dashboard import BenchmarkDashboardModel

        return BenchmarkDashboardModel.from_history(self.load()).model_dump(
            mode="json"
        )

    def _write(self, history: list[BenchmarkRunRecord]) -> None:
        """Persist history atomically enough for local demos."""
        payload = [r.model_dump(mode="json") for r in history]
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)


__all__ = [
    "BenchmarkRunRecord",
    "BenchmarkComparison",
    "BenchmarkHistory",
]
