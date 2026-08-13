"""Thin HTTP client for the FastAPI backend.

No RAG / agent / evaluation logic — only HTTP calls and error mapping.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ApiClientError(Exception):
    """Human-readable API client failure (safe to show in the UI)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ApiClient:
    """Minimal FastAPI client for the Streamlit presentation layer."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def close(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ApiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_health(self) -> dict[str, Any]:
        """GET /health."""
        return self._request("GET", "/health")

    def query_rag(
        self,
        question: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/v1/rag/query."""
        if not question or not question.strip():
            raise ApiClientError("Question cannot be empty.")
        payload: dict[str, Any] = {"question": question.strip()}
        if top_k is not None:
            payload["top_k"] = top_k
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        return self._request("POST", "/api/v1/rag/query", json=payload)

    def query_agent(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
        expected_answer: str | None = None,
        expected_sources: list[str] | None = None,
        validate_tools: bool = True,
    ) -> dict[str, Any]:
        """POST /api/v1/agent/query."""
        if not question or not question.strip():
            raise ApiClientError("Question cannot be empty.")
        payload: dict[str, Any] = {
            "question": question.strip(),
            "validate_tools": validate_tools,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if expected_answer is not None:
            payload["expected_answer"] = expected_answer
        if expected_sources is not None:
            payload["expected_sources"] = expected_sources
        return self._request("POST", "/api/v1/agent/query", json=payload)

    def verify_employee(self, employee_id: str) -> dict[str, Any]:
        """POST /api/v1/employees/verify — optional employee identity check.

        When the route is not yet deployed the UI falls back to a local demo
        ID check. This method only performs the HTTP call.
        """
        if not employee_id or not employee_id.strip():
            raise ApiClientError("Employee ID cannot be empty.", status_code=400)
        return self._request(
            "POST",
            "/api/v1/employees/verify",
            json={"employee_id": employee_id.strip()},
        )

    def run_demo(
        self,
        *,
        live: bool = False,
        output_dir: str = "./data/demo",
    ) -> dict[str, Any]:
        """POST /api/v1/demo/run — canonical agent-tools-foundation-001."""
        return self._request(
            "POST",
            "/api/v1/demo/run",
            json={"live": live, "output_dir": output_dir},
        )

    def list_reports(self, kind: str | None = None) -> dict[str, Any]:
        """GET /api/v1/reports."""
        params = {"kind": kind} if kind else None
        return self._request("GET", "/api/v1/reports", params=params)

    def get_report(self, kind: str, name: str) -> dict[str, Any]:
        """GET /api/v1/reports/content."""
        return self._request(
            "GET",
            "/api/v1/reports/content",
            params={"kind": kind, "name": name},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one HTTP request and map failures to ``ApiClientError``."""
        try:
            response = self._client.request(method, path, json=json, params=params)
        except httpx.ConnectError as exc:
            logger.error("API unavailable: base_url=%s error=%s", self._base_url, exc)
            raise ApiClientError(
                "Backend unavailable. "
                f"Expected API at {self._base_url}."
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("API timeout: path=%s timeout=%s", path, self._timeout)
            raise ApiClientError(
                f"Request timed out after {self._timeout:.0f}s. "
                "The backend may still be processing — try again."
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("HTTP error: path=%s error=%s", path, exc)
            raise ApiClientError("Unexpected network error talking to the API.") from exc

        if response.status_code >= 400:
            detail = _extract_detail(response)
            logger.warning(
                "API error: method=%s path=%s status=%s detail=%s",
                method,
                path,
                response.status_code,
                detail,
            )
            raise ApiClientError(detail, status_code=response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Malformed JSON from API: path=%s", path)
            raise ApiClientError(
                "Backend returned a malformed response (expected JSON)."
            ) from exc

        if not isinstance(payload, dict):
            raise ApiClientError(
                "Backend returned an unexpected response shape (expected object)."
            )
        return payload


def _extract_detail(response: httpx.Response) -> str:
    """Build a user-safe error message from an HTTP error response."""
    try:
        body = response.json()
    except ValueError:
        return f"Request failed with status {response.status_code}."

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail
    if isinstance(detail, list):
        # FastAPI validation errors
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(x) for x in item.get("loc", []) if x != "body")
                msg = item.get("msg", "invalid")
                parts.append(f"{loc}: {msg}" if loc else str(msg))
        if parts:
            return "Invalid request — " + "; ".join(parts)
    return f"Request failed with status {response.status_code}."
