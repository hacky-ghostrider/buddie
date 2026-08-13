"""Cost estimation helpers for evaluation reports."""

from __future__ import annotations

from typing import Any


def estimate_cost_usd(
    token_usage: dict[str, Any],
    *,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float | None:
    """Estimate USD cost from token usage and per-1K rates.

    Args:
        token_usage: Dict with ``prompt_tokens`` / ``completion_tokens``.
        input_cost_per_1k: USD per 1K prompt tokens.
        output_cost_per_1k: USD per 1K completion tokens.

    Returns:
        Estimated cost, or ``None`` when token counts are missing.
    """
    prompt = token_usage.get("prompt_tokens")
    completion = token_usage.get("completion_tokens")
    if prompt is None and completion is None:
        return None
    prompt_n = float(prompt or 0)
    completion_n = float(completion or 0)
    cost = (prompt_n / 1000.0) * input_cost_per_1k + (
        completion_n / 1000.0
    ) * output_cost_per_1k
    return round(cost, 8)


__all__ = ["estimate_cost_usd"]
