"""Streamlit UI configuration — no secrets, no business logic."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontendConfig:
    """Runtime config for the thin Streamlit client."""

    api_base_url: str
    request_timeout_seconds: float
    streamlit_port: int
    page_title: str
    page_icon: str


def load_config() -> FrontendConfig:
    """Load frontend settings from environment variables.

    Secrets (OpenAI, LangSmith) stay on the FastAPI backend only.
    """
    return FrontendConfig(
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        request_timeout_seconds=float(os.getenv("API_TIMEOUT_SECONDS", "120")),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
        page_title=os.getenv("STREAMLIT_PAGE_TITLE", "Buddie"),
        page_icon=os.getenv("STREAMLIT_PAGE_ICON", ":material/smart_toy:"),
    )
