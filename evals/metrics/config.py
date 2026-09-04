"""Explicit, easy-to-change DeepEval metric thresholds for Buddie evals.

Uses the same default pass cut-off as ``app.evaluation.deepeval`` adapters
(``0.7`` / ``Settings.default_pass_threshold``). LLM judge is Gemini via
DeepEval's native ``GeminiModel`` when a Gemini/Google API key is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Final


DEFAULT_PASS_THRESHOLD: Final[float] = 0.7
DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD: Final[float] = 0.35

# DeepEval-native Gemini Flash model suitable for evaluation judges.
# gemini-2.5-flash is unavailable to new API keys; flash-lite has higher free-tier RPM.
DEFAULT_GEMINI_JUDGE_MODEL: Final[str] = "gemini-3.1-flash-lite"

# Prefer DeepEval's GOOGLE_API_KEY; accept GEMINI_API_KEY as an alias.
_GEMINI_API_KEY_ENV_VARS: Final[tuple[str, ...]] = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)

METRIC_FAITHFULNESS: Final[str] = "faithfulness"
METRIC_ANSWER_RELEVANCY: Final[str] = "answer_relevancy"
METRIC_HALLUCINATION: Final[str] = "hallucination"
METRIC_CONTEXTUAL_PRECISION: Final[str] = "contextual_precision"
METRIC_CONTEXTUAL_RECALL: Final[str] = "contextual_recall"
METRIC_CONTEXTUAL_RELEVANCY: Final[str] = "contextual_relevancy"
METRIC_FINAL_RESPONSE_CORRECTNESS: Final[str] = "final_response_correctness"
METRIC_SEMANTIC_SIMILARITY: Final[str] = "semantic_similarity"

PRIMARY_METRIC_NAMES: Final[tuple[str, ...]] = (
    METRIC_FAITHFULNESS,
    METRIC_ANSWER_RELEVANCY,
    METRIC_HALLUCINATION,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_FINAL_RESPONSE_CORRECTNESS,
)

# Metrics that require non-empty actual runtime retrieval_context / context.
METRICS_REQUIRING_RETRIEVAL_CONTEXT: Final[frozenset[str]] = frozenset(
    {
        METRIC_FAITHFULNESS,
        METRIC_HALLUCINATION,
        METRIC_CONTEXTUAL_PRECISION,
        METRIC_CONTEXTUAL_RECALL,
        METRIC_CONTEXTUAL_RELEVANCY,
    }
)

# DeepEval HallucinationMetric is higher=worse; we invert to higher-is-better.
METRICS_INVERT_RAW_SCORE: Final[frozenset[str]] = frozenset(
    {METRIC_HALLUCINATION}
)

_ENV_LOADED: bool = False

# Bounded Gemini 429 retry defaults (overridable via env).
_DEFAULT_GEMINI_MAX_RETRIES: Final[int] = 3
_DEFAULT_GEMINI_RETRY_DELAY_SEC: Final[float] = 5.0


def _normalize_gemini_api_key_aliases() -> None:
    """Canonicalize Gemini API key to ``GOOGLE_API_KEY`` only.

    Accepts ``GEMINI_API_KEY`` as a legacy alias when ``GOOGLE_API_KEY`` is
    unset, then removes ``GEMINI_API_KEY`` from the process environment to
    avoid google-genai's dual-key warning.
    """
    google = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini = os.getenv("GEMINI_API_KEY", "").strip()
    canonical = google or gemini
    if canonical:
        os.environ["GOOGLE_API_KEY"] = canonical
    os.environ.pop("GEMINI_API_KEY", None)


def gemini_retry_max_retries() -> int:
    """Maximum *additional* attempts after the first 429 failure."""
    ensure_gemini_env_loaded()
    raw = os.getenv("EVAL_GEMINI_MAX_RETRIES", "").strip()
    if not raw:
        return _DEFAULT_GEMINI_MAX_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_GEMINI_MAX_RETRIES


def gemini_retry_default_delay_sec() -> float:
    """Fallback sleep when a 429 response omits an explicit retry delay."""
    ensure_gemini_env_loaded()
    raw = os.getenv("EVAL_GEMINI_RETRY_DELAY_SEC", "").strip()
    if not raw:
        return _DEFAULT_GEMINI_RETRY_DELAY_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_GEMINI_RETRY_DELAY_SEC


def ensure_gemini_env_loaded() -> None:
    """Load ``.env`` into ``os.environ`` and normalize Gemini API key aliases.

    Pydantic Settings only maps declared fields; ``GOOGLE_API_KEY`` /
    ``GEMINI_API_KEY`` are ignored there and must be loaded explicitly so
    DeepEval metrics receive a real ``GeminiModel`` instead of defaulting to
    ``GPTModel``.

    Prefer ``GOOGLE_API_KEY``; if only ``GEMINI_API_KEY`` is set, copy it to
    ``GOOGLE_API_KEY`` and unset ``GEMINI_API_KEY``. Never logs the secret.
    """
    global _ENV_LOADED
    if not _ENV_LOADED:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover — declared in pyproject
            pass
        _ENV_LOADED = True
    _normalize_gemini_api_key_aliases()


def _read_gemini_api_key() -> tuple[str | None, str | None]:
    """Return ``(api_key, env_var_name)`` without logging the secret."""
    ensure_gemini_env_loaded()
    # Legacy alias is normalized away; only GOOGLE_API_KEY remains in the process env.
    value = os.getenv("GOOGLE_API_KEY", "").strip()
    if value:
        return value, "GOOGLE_API_KEY"
    return None, None


def gemini_judge_configured() -> bool:
    """True when a Gemini/Google API key is present in the environment."""
    key, _ = _read_gemini_api_key()
    return key is not None


def gemini_judge_model_name() -> str:
    """Resolved Gemini judge model id (env override or default Flash)."""
    ensure_gemini_env_loaded()
    override = os.getenv("GEMINI_MODEL_NAME", "").strip()
    return override or DEFAULT_GEMINI_JUDGE_MODEL


def gemini_judge_status() -> dict[str, Any]:
    """Safe status dict for CLI/logging (never includes the API key)."""
    _, env_var = _read_gemini_api_key()
    configured = env_var is not None
    return {
        "configured": configured,
        "env_var": env_var,
        "model_name": gemini_judge_model_name() if configured else None,
        "provider": "Gemini",
    }


def resolve_deepeval_judge_model() -> Any | None:
    """Build DeepEval ``GeminiModel`` when an API key is available.

    Returns ``None`` when no key is configured. Does not print or return the
    secret. Callers that construct live LLM metrics must pass the returned
    object explicitly — never rely on DeepEval's default ``GPTModel``.
    """
    api_key, _ = _read_gemini_api_key()
    if not api_key:
        return None
    from deepeval.models import GeminiModel

    return GeminiModel(
        model=gemini_judge_model_name(),
        api_key=api_key,
        temperature=0,
    )


def require_deepeval_judge_model() -> Any:
    """Return a ``GeminiModel`` or raise if the Gemini API key is missing."""
    model = resolve_deepeval_judge_model()
    if model is None:
        raise RuntimeError(
            "Gemini LLM judge is not configured. Set GOOGLE_API_KEY or "
            "GEMINI_API_KEY so DeepEval metrics use GeminiModel instead of "
            "falling back to GPTModel / OPENAI_API_KEY."
        )
    return model


@dataclass(frozen=True)
class BuddieDeepEvalConfig:
    """Per-metric thresholds for the Buddie DeepEval / evaluation suite."""

    faithfulness: float = DEFAULT_PASS_THRESHOLD
    answer_relevancy: float = DEFAULT_PASS_THRESHOLD
    hallucination: float = DEFAULT_PASS_THRESHOLD
    contextual_precision: float = DEFAULT_PASS_THRESHOLD
    contextual_recall: float = DEFAULT_PASS_THRESHOLD
    contextual_relevancy: float = DEFAULT_PASS_THRESHOLD
    final_response_correctness: float = DEFAULT_PASS_THRESHOLD
    semantic_similarity: float = DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD
    # Case passes only when every non-skipped metric passes.
    require_all_metrics: bool = True
    # DeepEval judge: must be an explicit GeminiModel instance for live LLM metrics.
    model: Any | None = None
    metric_names: tuple[str, ...] = field(default_factory=lambda: PRIMARY_METRIC_NAMES)

    def threshold_for(self, metric_name: str) -> float:
        """Return the pass threshold for a known metric name."""
        mapping = {
            METRIC_FAITHFULNESS: self.faithfulness,
            METRIC_ANSWER_RELEVANCY: self.answer_relevancy,
            METRIC_HALLUCINATION: self.hallucination,
            METRIC_CONTEXTUAL_PRECISION: self.contextual_precision,
            METRIC_CONTEXTUAL_RECALL: self.contextual_recall,
            METRIC_CONTEXTUAL_RELEVANCY: self.contextual_relevancy,
            METRIC_FINAL_RESPONSE_CORRECTNESS: self.final_response_correctness,
            METRIC_SEMANTIC_SIMILARITY: self.semantic_similarity,
        }
        if metric_name not in mapping:
            raise KeyError(f"Unknown Buddie DeepEval metric: {metric_name}")
        return mapping[metric_name]


def default_buddie_deepeval_config() -> BuddieDeepEvalConfig:
    """Build config using project default pass threshold + Gemini judge when set."""
    threshold = DEFAULT_PASS_THRESHOLD
    try:
        from app.config.settings import get_settings

        threshold = float(get_settings().default_pass_threshold)
    except Exception:  # noqa: BLE001 — evals must still work offline
        pass

    return BuddieDeepEvalConfig(
        faithfulness=threshold,
        answer_relevancy=threshold,
        hallucination=threshold,
        contextual_precision=threshold,
        contextual_recall=threshold,
        contextual_relevancy=threshold,
        final_response_correctness=threshold,
        semantic_similarity=DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD,
        model=resolve_deepeval_judge_model(),
    )


__all__ = [
    "DEFAULT_GEMINI_JUDGE_MODEL",
    "DEFAULT_PASS_THRESHOLD",
    "DEFAULT_SEMANTIC_SIMILARITY_THRESHOLD",
    "METRIC_ANSWER_RELEVANCY",
    "METRIC_CONTEXTUAL_PRECISION",
    "METRIC_CONTEXTUAL_RECALL",
    "METRIC_CONTEXTUAL_RELEVANCY",
    "METRIC_FAITHFULNESS",
    "METRIC_FINAL_RESPONSE_CORRECTNESS",
    "METRIC_SEMANTIC_SIMILARITY",
    "METRIC_HALLUCINATION",
    "METRICS_INVERT_RAW_SCORE",
    "METRICS_REQUIRING_RETRIEVAL_CONTEXT",
    "PRIMARY_METRIC_NAMES",
    "BuddieDeepEvalConfig",
    "default_buddie_deepeval_config",
    "ensure_gemini_env_loaded",
    "gemini_judge_configured",
    "gemini_judge_model_name",
    "gemini_judge_status",
    "gemini_retry_default_delay_sec",
    "gemini_retry_max_retries",
    "require_deepeval_judge_model",
    "resolve_deepeval_judge_model",
]
