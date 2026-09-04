"""Unit tests: DeepEval LLM metrics must wire an explicit GeminiModel judge."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sanity

from evals.metrics.config import (
    METRIC_ANSWER_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_HALLUCINATION,
    BuddieDeepEvalConfig,
    default_buddie_deepeval_config,
    ensure_gemini_env_loaded,
    gemini_judge_configured,
    resolve_deepeval_judge_model,
)
from evals.metrics.g_eval import build_final_response_correctness_metric
from evals.metrics.standard import build_standard_llm_metrics


_FAKE_KEY = "fake-gemini-key-for-unit-tests-only"


@pytest.fixture
def gemini_env(monkeypatch: pytest.MonkeyPatch):
    """Isolate Gemini env vars with a fake key (never a real secret)."""
    monkeypatch.setenv("GOOGLE_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("GEMINI_MODEL_NAME", raising=False)
    ensure_gemini_env_loaded()
    return _FAKE_KEY


def test_resolve_judge_uses_gemini_model_not_gpt(gemini_env: str) -> None:
    pytest.importorskip("deepeval")
    from deepeval.models import GeminiModel, GPTModel

    model = resolve_deepeval_judge_model()
    assert model is not None
    assert isinstance(model, GeminiModel)
    assert not isinstance(model, GPTModel)
    assert gemini_judge_configured() is True


def test_gemini_api_key_alias_normalizes_to_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", _FAKE_KEY)
    ensure_gemini_env_loaded()
    assert gemini_judge_configured() is True
    import os

    assert os.environ.get("GOOGLE_API_KEY") == _FAKE_KEY
    assert "GEMINI_API_KEY" not in os.environ


def test_all_llm_metrics_receive_gemini_model(gemini_env: str) -> None:
    pytest.importorskip("deepeval")
    from deepeval.models import GeminiModel, GPTModel

    config = default_buddie_deepeval_config()
    assert isinstance(config.model, GeminiModel)
    assert not isinstance(config.model, GPTModel)

    standard = build_standard_llm_metrics(config)
    expected = {
        METRIC_FAITHFULNESS,
        METRIC_ANSWER_RELEVANCY,
        METRIC_HALLUCINATION,
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    }
    assert set(standard) == expected
    for name, metric in standard.items():
        assert isinstance(metric.model, GeminiModel), name
        assert not isinstance(metric.model, GPTModel), name
        # Same explicit judge instance configuration (shared object).
        assert metric.model is config.model

    g_eval = build_final_response_correctness_metric(
        config.threshold_for("final_response_correctness"),
        model=config.model,
    )
    assert isinstance(g_eval.model, GeminiModel)
    assert not isinstance(g_eval.model, GPTModel)
    assert g_eval.model is config.model


def test_live_metric_rejects_missing_model() -> None:
    pytest.importorskip("deepeval")
    from evals.metrics.standard import _live_metric

    with pytest.raises(RuntimeError, match="GeminiModel"):
        _live_metric(METRIC_FAITHFULNESS, 0.7, None)


def test_geval_rejects_missing_model() -> None:
    pytest.importorskip("deepeval")
    with pytest.raises(RuntimeError, match="GeminiModel"):
        build_final_response_correctness_metric(0.7, model=None)


def test_explicit_config_model_propagates(gemini_env: str) -> None:
    pytest.importorskip("deepeval")
    from deepeval.models import GeminiModel

    judge = resolve_deepeval_judge_model()
    config = BuddieDeepEvalConfig(model=judge)
    metrics = build_standard_llm_metrics(config)
    assert all(isinstance(m.model, GeminiModel) for m in metrics.values())
