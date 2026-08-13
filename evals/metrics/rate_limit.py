"""Bounded retry handling for Gemini free-tier 429 / RESOURCE_EXHAUSTED errors."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

from evals.metrics.config import (
    gemini_retry_default_delay_sec,
    gemini_retry_max_retries,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_IN_SECONDS_RE = re.compile(
    r"retry in ([\d.]+)s",
    re.IGNORECASE,
)
_RETRY_DELAY_FIELD_RE = re.compile(
    r"['\"]retryDelay['\"]\s*:\s*['\"](\d+)s['\"]",
    re.IGNORECASE,
)


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when an exception looks like a Gemini/API quota or rate-limit failure."""
    text = str(exc)
    upper = text.upper()
    return (
        "429" in text
        or "RESOURCE_EXHAUSTED" in upper
        or "RATE LIMIT" in upper
        or "QUOTA" in upper
    )


def parse_retry_delay_sec(exc: BaseException) -> float | None:
    """Extract server-suggested retry delay from a Gemini 429 error message."""
    text = str(exc)
    match = _RETRY_IN_SECONDS_RE.search(text)
    if match:
        return max(0.0, float(match.group(1)))
    delay_match = _RETRY_DELAY_FIELD_RE.search(text)
    if delay_match:
        return max(0.0, float(delay_match.group(1)))
    return None


def call_with_gemini_rate_limit_retry(
    fn: Callable[[], T],
    *,
    max_retries: int | None = None,
    default_delay_sec: float | None = None,
) -> T:
    """Call ``fn`` with bounded retries when Gemini returns 429 / quota errors.

  ``max_retries`` is the number of *additional* attempts after the first failure
  (total attempts = ``max_retries + 1``). Never retries indefinitely.
    """
    retries_allowed = (
        gemini_retry_max_retries() if max_retries is None else max(0, max_retries)
    )
    fallback_delay = (
        gemini_retry_default_delay_sec()
        if default_delay_sec is None
        else max(0.0, default_delay_sec)
    )
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classify vendor rate limits
            if not is_rate_limit_error(exc) or attempt >= retries_allowed:
                raise
            delay = parse_retry_delay_sec(exc)
            if delay is None:
                delay = fallback_delay
            logger.warning(
                "Gemini rate limit on attempt %s/%s; sleeping %.1fs before retry",
                attempt + 1,
                retries_allowed + 1,
                delay,
            )
            time.sleep(delay)
            attempt += 1


__all__ = [
    "call_with_gemini_rate_limit_retry",
    "is_rate_limit_error",
    "parse_retry_delay_sec",
]
