"""Custom G-Eval: FINAL_RESPONSE_CORRECTNESS (Sprint 18C).

Uses golden expected_answer as the evaluation reference.
Does not judge deterministic tool selection (pytest territory).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from evals.metrics.config import (
    METRIC_FINAL_RESPONSE_CORRECTNESS,
    BuddieDeepEvalConfig,
)
from evals.metrics.rate_limit import (
    call_with_gemini_rate_limit_retry,
    is_rate_limit_error,
)
from evals.metrics.results import MetricScoreResult
from evals.metrics.standard import MetricMeasureFn

if TYPE_CHECKING:
    from evals.runners.deepeval_case import DeepEvalCompatibleCase

logger = logging.getLogger(__name__)

FINAL_RESPONSE_CORRECTNESS_CRITERIA = (
    "Evaluate whether the actual response correctly and completely answers "
    "the user's query, using the expected output as the reference answer. "
    "Check: (1) Does the response correctly answer the user's query? "
    "(2) Is it consistent with the supplied retrieval/evidence context when "
    "present? (3) Does it avoid unsupported claims? "
    "(4) Does it satisfy the expected behavior implied by the reference? "
    "(5) Is it complete enough for the question? "
    "(6) Does it avoid irrelevant information? "
    "Tool-selection correctness is out of scope for this metric."
)

FINAL_RESPONSE_CORRECTNESS_STEPS = [
    "Read the user input and the expected output (reference).",
    "Compare the actual output to the reference for correctness and completeness.",
    "If retrieval_context is present, check consistency and unsupported claims.",
    "Penalize irrelevant content and missing required information.",
    "Score higher when the response matches expected behavior without fabricating facts.",
]


def _evaluation_params() -> list[Any]:
    try:
        from deepeval.test_case import LLMTestCaseParams

        return [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ]
    except ImportError:  # pragma: no cover
        from deepeval.test_case import SingleTurnParams

        return [
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ]


def build_final_response_correctness_metric(
    threshold: float,
    *,
    model: Any | None = None,
) -> Any:
    """Construct DeepEval GEval for FINAL_RESPONSE_CORRECTNESS.

    ``model`` must be an explicit DeepEval ``GeminiModel`` instance. Omitting it
    would silently fall back to ``GPTModel`` / ``OPENAI_API_KEY``.
    """
    from deepeval.metrics import GEval

    if model is None:
        raise RuntimeError(
            "GEval FINAL_RESPONSE_CORRECTNESS requires an explicit GeminiModel. "
            "Set GOOGLE_API_KEY or GEMINI_API_KEY and pass config.model."
        )

    return GEval(
        name="FINAL_RESPONSE_CORRECTNESS",
        criteria=FINAL_RESPONSE_CORRECTNESS_CRITERIA,
        evaluation_steps=FINAL_RESPONSE_CORRECTNESS_STEPS,
        evaluation_params=_evaluation_params(),
        threshold=threshold,
        model=model,
    )


def _rate_limited(threshold: float, detail: str) -> MetricScoreResult:
    return MetricScoreResult(
        name=METRIC_FINAL_RESPONSE_CORRECTNESS,
        score=None,
        passed=None,
        threshold=threshold,
        rate_limited=True,
        error=f"RATE_LIMITED: {detail}",
    )


def measure_final_response_correctness(
    case: DeepEvalCompatibleCase,
    config: BuddieDeepEvalConfig,
    *,
    measure_fn: MetricMeasureFn | None = None,
) -> MetricScoreResult:
    """Score FINAL_RESPONSE_CORRECTNESS via G-Eval (or injectable measure_fn)."""
    threshold = config.threshold_for(METRIC_FINAL_RESPONSE_CORRECTNESS)

    if measure_fn is not None:
        test_case = type("SimpleTestCase", (), case.to_llm_test_case_kwargs())()
        return measure_fn(
            METRIC_FINAL_RESPONSE_CORRECTNESS,
            test_case,
            threshold=threshold,
        )

    started = time.perf_counter()
    try:
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(**case.to_llm_test_case_kwargs())
        metric = build_final_response_correctness_metric(
            threshold, model=config.model
        )

        def _measure() -> None:
            metric.measure(test_case)

        call_with_gemini_rate_limit_retry(_measure)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_score = getattr(metric, "score", None)
        success = getattr(metric, "success", None)
        reason = getattr(metric, "reason", None)
        score = float(raw_score) if raw_score is not None else None
        passed = bool(success) if success is not None else (
            score is not None and score >= threshold
        )
        logger.debug(
            "G-Eval FINAL_RESPONSE_CORRECTNESS case=%s score=%s latency_ms=%.1f",
            case.case_id,
            score,
            latency_ms,
        )
        return MetricScoreResult(
            name=METRIC_FINAL_RESPONSE_CORRECTNESS,
            score=round(score, 6) if score is not None else None,
            passed=passed,
            threshold=threshold,
            reason=str(reason) if reason else None,
        )
    except Exception as exc:  # noqa: BLE001
        if is_rate_limit_error(exc):
            logger.warning(
                "G-Eval rate limited: case=%s",
                case.case_id,
            )
            return _rate_limited(threshold, str(exc))
        logger.exception(
            "G-Eval failed: case=%s",
            case.case_id,
        )
        return MetricScoreResult(
            name=METRIC_FINAL_RESPONSE_CORRECTNESS,
            score=None,
            passed=False,
            threshold=threshold,
            error=f"{METRIC_FINAL_RESPONSE_CORRECTNESS} failed: {exc}",
        )


__all__ = [
    "FINAL_RESPONSE_CORRECTNESS_CRITERIA",
    "FINAL_RESPONSE_CORRECTNESS_STEPS",
    "build_final_response_correctness_metric",
    "measure_final_response_correctness",
]
