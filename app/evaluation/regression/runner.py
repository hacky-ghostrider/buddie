"""Regression comparison — detect quality / latency / tool / prompt drifts.

WHY
---
A single evaluation score is a snapshot. Regression testing compares
*current* vs *previous* runs so prompt edits, model upgrades, or retrieval
tweaks cannot silently degrade quality — the AI analogue of comparing
two JUnit XML result sets across builds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.exceptions import InvalidEvaluationInputError
from app.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)


class MetricRegression(BaseModel):
    """Per-metric score delta for one question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    metric_name: str
    previous_score: float
    current_score: float
    delta: float
    is_regression: bool


class LatencyRegression(BaseModel):
    """Latency delta for one question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    previous_latency_ms: float
    current_latency_ms: float
    delta_ms: float
    is_regression: bool


class ToolRegression(BaseModel):
    """Tool-validation pass/fail change for one question."""

    model_config = ConfigDict(extra="forbid")

    question: str
    previous_passed: bool | None
    current_passed: bool | None
    previous_tools: list[str] = Field(default_factory=list)
    current_tools: list[str] = Field(default_factory=list)
    is_regression: bool


class PromptRegression(BaseModel):
    """Heuristic prompt drift signal via answer / metadata changes."""

    model_config = ConfigDict(extra="forbid")

    question: str
    previous_answer: str
    current_answer: str
    answer_changed: bool
    previous_model: str | None = None
    current_model: str | None = None
    is_regression: bool


class CostRegression(BaseModel):
    """Estimated cost delta for one question (Sprint 12)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    previous_cost_usd: float | None = None
    current_cost_usd: float | None = None
    delta_usd: float | None = None
    is_regression: bool = False


class RegressionReport(BaseModel):
    """Aggregated regression comparison between two evaluation runs."""

    model_config = ConfigDict(extra="forbid")

    previous_path: str
    current_path: str
    score_regressions: list[MetricRegression] = Field(default_factory=list)
    latency_regressions: list[LatencyRegression] = Field(default_factory=list)
    tool_regressions: list[ToolRegression] = Field(default_factory=list)
    prompt_regressions: list[PromptRegression] = Field(default_factory=list)
    cost_regressions: list[CostRegression] = Field(default_factory=list)
    matched_questions: int = 0
    has_regressions: bool = False
    summary: dict[str, Any] = Field(default_factory=dict)


class RegressionRunner:
    """Compare previous vs current evaluation report artifacts.

    Args:
        score_drop_threshold: Absolute drop that counts as a metric regression.
        latency_increase_ratio: Relative latency increase that counts as
            regression (e.g. ``0.25`` = +25%).
        treat_answer_change_as_prompt_regression: Flag answer text changes.
    """

    def __init__(
        self,
        *,
        score_drop_threshold: float = 0.05,
        latency_increase_ratio: float = 0.25,
        cost_increase_ratio: float = 0.25,
        treat_answer_change_as_prompt_regression: bool = True,
    ) -> None:
        if score_drop_threshold < 0:
            raise ValueError("score_drop_threshold must be >= 0")
        if latency_increase_ratio < 0:
            raise ValueError("latency_increase_ratio must be >= 0")
        if cost_increase_ratio < 0:
            raise ValueError("cost_increase_ratio must be >= 0")
        self._score_drop_threshold = score_drop_threshold
        self._latency_increase_ratio = latency_increase_ratio
        self._cost_increase_ratio = cost_increase_ratio
        self._answer_change = treat_answer_change_as_prompt_regression

    def compare_files(
        self,
        previous_path: str | Path,
        current_path: str | Path,
    ) -> RegressionReport:
        """Load two JSON report files and compare them."""
        previous = self._load_reports(previous_path)
        current = self._load_reports(current_path)
        return self.compare(
            previous,
            current,
            previous_path=str(previous_path),
            current_path=str(current_path),
        )

    def compare(
        self,
        previous: list[EvaluationReport],
        current: list[EvaluationReport],
        *,
        previous_path: str = "",
        current_path: str = "",
    ) -> RegressionReport:
        """Compare in-memory report lists keyed by question text."""
        prev_by_q = {r.question: r for r in previous}
        curr_by_q = {r.question: r for r in current}
        shared = sorted(set(prev_by_q) & set(curr_by_q))

        score_regs: list[MetricRegression] = []
        latency_regs: list[LatencyRegression] = []
        tool_regs: list[ToolRegression] = []
        prompt_regs: list[PromptRegression] = []
        cost_regs: list[CostRegression] = []

        for question in shared:
            prev = prev_by_q[question]
            curr = curr_by_q[question]
            score_regs.extend(self._score_deltas(question, prev, curr))
            latency_regs.append(self._latency_delta(question, prev, curr))
            tool_regs.append(self._tool_delta(question, prev, curr))
            prompt_regs.append(self._prompt_delta(question, prev, curr))
            cost_regs.append(self._cost_delta(question, prev, curr))

        score_hits = [r for r in score_regs if r.is_regression]
        latency_hits = [r for r in latency_regs if r.is_regression]
        tool_hits = [r for r in tool_regs if r.is_regression]
        prompt_hits = [r for r in prompt_regs if r.is_regression]
        cost_hits = [r for r in cost_regs if r.is_regression]
        has_regs = bool(
            score_hits or latency_hits or tool_hits or prompt_hits or cost_hits
        )

        report = RegressionReport(
            previous_path=previous_path,
            current_path=current_path,
            score_regressions=score_hits,
            latency_regressions=latency_hits,
            tool_regressions=tool_hits,
            prompt_regressions=prompt_hits,
            cost_regressions=cost_hits,
            matched_questions=len(shared),
            has_regressions=has_regs,
            summary={
                "score_regression_count": len(score_hits),
                "latency_regression_count": len(latency_hits),
                "tool_regression_count": len(tool_hits),
                "prompt_regression_count": len(prompt_hits),
                "cost_regression_count": len(cost_hits),
                "unmatched_previous": sorted(set(prev_by_q) - set(curr_by_q)),
                "unmatched_current": sorted(set(curr_by_q) - set(prev_by_q)),
            },
        )
        logger.info(
            "Regression comparison: matched=%d has_regressions=%s summary=%s",
            report.matched_questions,
            report.has_regressions,
            report.summary,
        )
        return report

    def _score_deltas(
        self,
        question: str,
        previous: EvaluationReport,
        current: EvaluationReport,
    ) -> list[MetricRegression]:
        prev_scores = {m.name: m.score for m in previous.metrics}
        curr_scores = {m.name: m.score for m in current.metrics}
        names = sorted(set(prev_scores) & set(curr_scores))
        results: list[MetricRegression] = []
        for name in names:
            prev_s = prev_scores[name]
            curr_s = curr_scores[name]
            delta = curr_s - prev_s
            results.append(
                MetricRegression(
                    question=question,
                    metric_name=name,
                    previous_score=prev_s,
                    current_score=curr_s,
                    delta=round(delta, 6),
                    is_regression=delta <= -self._score_drop_threshold,
                )
            )
        # Overall score regression
        overall_delta = current.overall_score - previous.overall_score
        results.append(
            MetricRegression(
                question=question,
                metric_name="overall_score",
                previous_score=previous.overall_score,
                current_score=current.overall_score,
                delta=round(overall_delta, 6),
                is_regression=overall_delta <= -self._score_drop_threshold,
            )
        )
        return results

    def _latency_delta(
        self,
        question: str,
        previous: EvaluationReport,
        current: EvaluationReport,
    ) -> LatencyRegression:
        prev_l = (
            previous.rag_latency_ms
            if previous.rag_latency_ms is not None
            else previous.latency
        )
        curr_l = (
            current.rag_latency_ms
            if current.rag_latency_ms is not None
            else current.latency
        )
        delta = curr_l - prev_l
        ratio_hit = False
        if prev_l > 0:
            ratio_hit = (curr_l / prev_l) - 1.0 >= self._latency_increase_ratio
        elif curr_l > prev_l:
            ratio_hit = True
        return LatencyRegression(
            question=question,
            previous_latency_ms=prev_l,
            current_latency_ms=curr_l,
            delta_ms=round(delta, 3),
            is_regression=ratio_hit,
        )

    def _tool_delta(
        self,
        question: str,
        previous: EvaluationReport,
        current: EvaluationReport,
    ) -> ToolRegression:
        prev_passed = (
            None if previous.tool_validation is None else previous.tool_validation.passed
        )
        curr_passed = (
            None if current.tool_validation is None else current.tool_validation.passed
        )
        prev_tools = (
            []
            if previous.tool_validation is None
            else list(previous.tool_validation.actual_tools)
        )
        curr_tools = (
            []
            if current.tool_validation is None
            else list(current.tool_validation.actual_tools)
        )
        is_reg = prev_passed is True and curr_passed is False
        return ToolRegression(
            question=question,
            previous_passed=prev_passed,
            current_passed=curr_passed,
            previous_tools=prev_tools,
            current_tools=curr_tools,
            is_regression=is_reg,
        )

    def _prompt_delta(
        self,
        question: str,
        previous: EvaluationReport,
        current: EvaluationReport,
    ) -> PromptRegression:
        prev_model = previous.metadata.get("model") or previous.token_usage.get("model")
        curr_model = current.metadata.get("model") or current.token_usage.get("model")
        answer_changed = previous.answer.strip() != current.answer.strip()
        is_reg = bool(self._answer_change and answer_changed)
        return PromptRegression(
            question=question,
            previous_answer=previous.answer,
            current_answer=current.answer,
            answer_changed=answer_changed,
            previous_model=str(prev_model) if prev_model else None,
            current_model=str(curr_model) if curr_model else None,
            is_regression=is_reg,
        )

    def _cost_delta(
        self,
        question: str,
        previous: EvaluationReport,
        current: EvaluationReport,
    ) -> CostRegression:
        """Detect cost regressions (relative increase vs previous run)."""
        prev_c = previous.estimated_cost_usd
        curr_c = current.estimated_cost_usd
        if prev_c is None or curr_c is None:
            return CostRegression(
                question=question,
                previous_cost_usd=prev_c,
                current_cost_usd=curr_c,
                delta_usd=None,
                is_regression=False,
            )
        delta = curr_c - prev_c
        ratio_hit = False
        if prev_c > 0:
            ratio_hit = (curr_c / prev_c) - 1.0 >= self._cost_increase_ratio
        elif curr_c > prev_c:
            ratio_hit = True
        return CostRegression(
            question=question,
            previous_cost_usd=prev_c,
            current_cost_usd=curr_c,
            delta_usd=round(delta, 8),
            is_regression=ratio_hit,
        )

    @staticmethod
    def _load_reports(path: str | Path) -> list[EvaluationReport]:
        """Load evaluation reports from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise InvalidEvaluationInputError(
                f"Evaluation report file not found: {file_path}"
            )
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise InvalidEvaluationInputError(
                "Regression inputs must be JSON arrays of EvaluationReport"
            )
        return [EvaluationReport.model_validate(item) for item in raw]


__all__ = [
    "MetricRegression",
    "LatencyRegression",
    "ToolRegression",
    "PromptRegression",
    "CostRegression",
    "RegressionReport",
    "RegressionRunner",
]
