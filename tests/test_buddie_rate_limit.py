"""Deterministic tests for Gemini 429 retry, rate-limited reporting, and applicability."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from evals.metrics.applicability import (
    has_annotated_expected_context,
    has_meaningful_retrieval_context,
    llm_metric_skip_reason,
)
from evals.metrics.config import (
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FAITHFULNESS,
    BuddieDeepEvalConfig,
    ensure_gemini_env_loaded,
    gemini_retry_default_delay_sec,
    gemini_retry_max_retries,
)
from evals.metrics.rate_limit import (
    call_with_gemini_rate_limit_retry,
    is_rate_limit_error,
    parse_retry_delay_sec,
)
from evals.metrics.results import MetricScoreResult
from evals.metrics.standard import measure_standard_metric
from evals.runners.deepeval_case import DeepEvalCompatibleCase
from evals.runners.deepeval_suite import evaluate_deepeval_case

_RATE_LIMIT_MSG = (
    "429 RESOURCE_EXHAUSTED. Please retry in 2.5s. "
    "quota exceeded for generativelanguage.googleapis.com"
)


class _RateLimitError(Exception):
    pass


def test_is_rate_limit_error_detects_429() -> None:
    assert is_rate_limit_error(_RateLimitError(_RATE_LIMIT_MSG)) is True
    assert is_rate_limit_error(RuntimeError("connection reset")) is False


def test_parse_retry_delay_from_message() -> None:
    assert parse_retry_delay_sec(_RateLimitError(_RATE_LIMIT_MSG)) == 2.5
    assert parse_retry_delay_sec(RuntimeError("no delay here")) is None


def test_call_with_rate_limit_retries_then_succeeds() -> None:
    calls = {"count": 0}

    def _flaky() -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise _RateLimitError(_RATE_LIMIT_MSG)
        return "ok"

    with patch("evals.metrics.rate_limit.time.sleep") as sleep_mock:
        result = call_with_gemini_rate_limit_retry(
            _flaky,
            max_retries=2,
            default_delay_sec=1.0,
        )
    assert result == "ok"
    assert calls["count"] == 2
    sleep_mock.assert_called_once_with(2.5)


def test_call_with_rate_limit_exhausts_retries() -> None:
    def _always_limit() -> None:
        raise _RateLimitError(_RATE_LIMIT_MSG)

    with patch("evals.metrics.rate_limit.time.sleep"):
        with pytest.raises(_RateLimitError):
            call_with_gemini_rate_limit_retry(
                _always_limit,
                max_retries=1,
                default_delay_sec=0.1,
            )


def test_rate_limited_metric_never_passes() -> None:
    def _rate_limited_measure(
        metric_name: str,
        test_case: Any,
        *,
        threshold: float,
    ) -> MetricScoreResult:
        del test_case
        return MetricScoreResult(
            name=metric_name,
            score=None,
            passed=None,
            threshold=threshold,
            rate_limited=True,
            error="RATE_LIMITED: quota exceeded",
        )

    case = DeepEvalCompatibleCase(
        case_id="rate-001",
        input="How much vacation?",
        actual_output="14 days",
        expected_output="14 days",
        retrieval_context=["evidence"],
        expected_context=["evidence"],
    )
    config = BuddieDeepEvalConfig()
    result = evaluate_deepeval_case(
        case,
        config,
        measure_fn=_rate_limited_measure,
    )
    for metric in result.metric_map().values():
        if not metric.skipped:
            assert metric.rate_limited is True
            assert metric.passed is None
            assert metric.outcome == "rate_limited"
    assert result.overall_status == "rate_limited"


def test_threshold_failure_takes_precedence_over_rate_limit() -> None:
    def _mixed_measure(
        metric_name: str,
        test_case: Any,
        *,
        threshold: float,
    ) -> MetricScoreResult:
        del test_case
        if metric_name == METRIC_FAITHFULNESS:
            return MetricScoreResult(
                name=metric_name,
                score=0.1,
                passed=False,
                threshold=threshold,
                reason="low faithfulness",
            )
        return MetricScoreResult(
            name=metric_name,
            score=None,
            passed=None,
            threshold=threshold,
            rate_limited=True,
            error="RATE_LIMITED: quota exceeded",
        )

    case = DeepEvalCompatibleCase(
        case_id="mixed-001",
        input="How much vacation?",
        actual_output="14 days",
        expected_output="14 days",
        retrieval_context=["evidence"],
        expected_context=["evidence"],
    )
    result = evaluate_deepeval_case(
        case,
        BuddieDeepEvalConfig(),
        measure_fn=_mixed_measure,
    )
    assert result.overall_status == "failed"
    assert result.faithfulness.passed is False


def test_contextual_metrics_skip_without_expected_context() -> None:
    case = DeepEvalCompatibleCase(
        case_id="tool-only",
        input="How much vacation?",
        actual_output="14 days",
        expected_output="14 days",
        retrieval_context=["get_leave_balance: {\"vacation\": 14}"],
        expected_context=[],
        expected_behavior="answer_from_tool",
    )
    assert has_meaningful_retrieval_context(case) is True
    assert has_annotated_expected_context(case) is False
    for metric in (
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    ):
        reason = llm_metric_skip_reason(metric, case)
        assert reason is not None
        assert "expected_context" in reason

    config = BuddieDeepEvalConfig()

    def _passing(
        metric_name: str,
        test_case: Any,
        *,
        threshold: float,
    ) -> MetricScoreResult:
        del test_case
        return MetricScoreResult(
            name=metric_name,
            score=0.95,
            passed=True,
            threshold=threshold,
        )

    result = evaluate_deepeval_case(case, config, measure_fn=_passing)
    assert result.contextual_precision.skipped is True
    assert result.contextual_recall.skipped is True
    assert result.contextual_relevancy is not None
    assert result.contextual_relevancy.skipped is True
    assert result.faithfulness.skipped is False
    assert result.faithfulness.passed is True


def test_measure_standard_metric_maps_exhausted_429_to_rate_limited() -> None:
    case = DeepEvalCompatibleCase(
        case_id="live-rate",
        input="Q",
        actual_output="A",
        expected_output="A",
        retrieval_context=["ctx"],
        expected_context=["ctx"],
    )
    config = BuddieDeepEvalConfig(model=object())
    dummy_metric = type("Metric", (), {"measure": lambda self, _tc: None})()

    with patch("evals.metrics.standard._live_metric", return_value=dummy_metric):
        with patch(
            "evals.metrics.standard.call_with_gemini_rate_limit_retry",
            side_effect=_RateLimitError(_RATE_LIMIT_MSG),
        ):
            result = measure_standard_metric(
                METRIC_FAITHFULNESS,
                case,
                config,
            )
    assert result.rate_limited is True
    assert result.passed is None
    assert result.outcome == "rate_limited"


def test_google_api_key_is_canonical_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    ensure_gemini_env_loaded()
    assert os.environ.get("GOOGLE_API_KEY") == "fake-google-key"
    assert "GEMINI_API_KEY" not in os.environ


def test_gemini_alias_copied_to_google_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-alias-key")
    ensure_gemini_env_loaded()
    assert os.environ.get("GOOGLE_API_KEY") == "legacy-alias-key"
    assert "GEMINI_API_KEY" not in os.environ


def test_retry_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_GEMINI_MAX_RETRIES", "5")
    monkeypatch.setenv("EVAL_GEMINI_RETRY_DELAY_SEC", "7.5")
    assert gemini_retry_max_retries() == 5
    assert gemini_retry_default_delay_sec() == 7.5
