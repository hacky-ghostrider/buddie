"""Unit tests for Buddie-facing error copy and verification helpers."""

from __future__ import annotations

import json

import httpx
import pytest

from frontend.api_client import ApiClient, ApiClientError
from frontend.components.quick_actions import PRIMARY_ACTIONS, SECONDARY_ACTIONS
from frontend.components.verification import (
    needs_employee_verification,
    normalize_employee_id,
    verify_employee_id,
)
from frontend.user_messages import friendly_error, technical_detail


def _client(handler) -> ApiClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="http://test", transport=transport)
    api = ApiClient("http://test", client=http)
    api._owns_client = True
    return api


def test_needs_employee_verification_for_personal_leave() -> None:
    assert needs_employee_verification("How many vacation days do I have left?")
    assert needs_employee_verification("What are my pending tasks?")
    assert not needs_employee_verification("What is the leave policy?")
    assert not needs_employee_verification("Upcoming company holidays?")


def test_normalize_employee_id() -> None:
    assert normalize_employee_id(" e-1101 ") == "E-1101"
    assert normalize_employee_id("1101") == "E-1101"
    # Incomplete / non-pattern digits must not become fake employee ids.
    assert normalize_employee_id("123") == "123"
    assert normalize_employee_id("12345") == "12345"


def test_needs_employee_verification_ignores_standalone_numbers() -> None:
    assert not needs_employee_verification("123")
    assert not needs_employee_verification("1101")
    assert not needs_employee_verification("E-1101")
    assert not needs_employee_verification("!!!")
    assert not needs_employee_verification("asdfgh")
    assert needs_employee_verification("How many vacation days do I have left?")


def test_looks_like_employee_id() -> None:
    from frontend.components.verification import looks_like_employee_id

    assert looks_like_employee_id("E-1101")
    assert looks_like_employee_id("e-1101")
    assert not looks_like_employee_id("123")
    assert not looks_like_employee_id("1101")
    assert not looks_like_employee_id("E-123")
    assert not looks_like_employee_id("456")
    assert not looks_like_employee_id("999")


def test_verify_employee_falls_back_when_route_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/employees/verify"
        return httpx.Response(404, json={"detail": "Not Found"})

    with _client(handler) as client:
        result = verify_employee_id(client, "E-1101")
    assert result["verified"] is True
    assert result["employee_id"] == "E-1101"


def test_verify_employee_falls_back_when_backend_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client:
        result = verify_employee_id(client, "E-1101")
    assert result["verified"] is True
    assert result["employee_id"] == "E-1101"


def test_verify_employee_rejects_unknown_demo_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    with _client(handler) as client:
        with pytest.raises(ApiClientError) as exc:
            verify_employee_id(client, "E-9999")
    assert "couldn't be verified" in exc.value.message.lower()


def test_verify_employee_uses_backend_when_available() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["employee_id"] == "E-1102"
        return httpx.Response(
            200,
            json={"verified": True, "employee_id": "E-1102", "name": "Alex"},
        )

    with _client(handler) as client:
        result = verify_employee_id(client, "E-1102")
    assert result["name"] == "Alex"


def test_friendly_error_hides_collection_internals() -> None:
    exc = ApiClientError(
        "Retrieval failed: Collection 'rag_documents' does not exist",
        status_code=502,
    )
    message = friendly_error(exc)
    assert "rag_documents" not in message
    assert "knowledge base" in message.lower()
    assert "rag_documents" in technical_detail(exc)


def test_primary_quick_actions_stay_small() -> None:
    assert len(PRIMARY_ACTIONS) == 4
    assert "My leave" in PRIMARY_ACTIONS
    assert "Leave history" in PRIMARY_ACTIONS
    assert "Upcoming holidays" in PRIMARY_ACTIONS
    assert "Pending actions" in PRIMARY_ACTIONS
    # Benefits / policies remain available, but not as the main grid.
    assert "My benefits" in SECONDARY_ACTIONS
    assert "Company policies" in SECONDARY_ACTIONS
    assert not set(PRIMARY_ACTIONS).intersection(SECONDARY_ACTIONS)
