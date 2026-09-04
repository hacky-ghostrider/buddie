"""Smoke tests for the Sprint 1 foundation (health endpoint and settings)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.main import create_app

pytestmark = pytest.mark.smoke


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint should return 200 with expected payload fields."""
    get_settings.cache_clear()
    settings = Settings(app_name="rag-evaluation-platform", app_env="test")
    client = TestClient(create_app(settings=settings))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "rag-evaluation-platform"
    assert payload["environment"] == "test"
    assert "version" in payload


def test_settings_loads_defaults() -> None:
    """Settings should provide sensible defaults without a .env file."""
    get_settings.cache_clear()
    settings = Settings()

    assert settings.app_name == "rag-evaluation-platform"
    assert settings.port == 8000
    assert settings.api_prefix == "/api/v1"
