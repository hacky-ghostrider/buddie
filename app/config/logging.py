"""Centralized logging configuration for the application.

Configures the root logger once at startup so every module can use
``logging.getLogger(__name__)`` without ad-hoc setup — analogous to a
Logback ``logback.xml`` / Log4j configuration.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging.

    Idempotent for typical startup use: replaces root handlers so repeated
    calls (e.g. tests) do not stack duplicate handlers.

    Args:
        level: Logging level name (e.g. ``INFO``, ``DEBUG``). Invalid names
            fall back to ``INFO``.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called more than once.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers in development unless DEBUG is needed.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
