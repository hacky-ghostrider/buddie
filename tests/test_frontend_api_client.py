"""Unit tests for the Streamlit API client (no live OpenAI / LangSmith)."""

from __future__ import annotations

import json

import httpx
import pytest

from frontend.api_client import ApiClient, ApiClientError


def _client(handler) -> ApiClient:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(base_url="http://test", transport=transport)
    api = ApiClient("http://test", client=http)
    api._owns_client = True
    return api


def test_query_rag_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v1/rag/query"
        body = json.loads(request.content.decode())
        assert body["question"] == "What is PTO?"
        return httpx.Response(
            200,
            json={
                "question": "What is PTO?",
                "answer": "Paid time off.",
                "retrieved_documents": [],
                "retrieval_metadata": {},
                "generation_metadata": {"model": "test"},
                "latency": {
                    "retrieval_ms": 1.0,
                    "prompt_build_ms": 1.0,
                    "llm_ms": 1.0,
                    "total_ms": 3.0,
                },
                "correlation_id": "c-1",
            },
        )

    with _client(handler) as client:
        result = client.query_rag("What is PTO?")
    assert result["answer"] == "Paid time off."
    assert result["correlation_id"] == "c-1"


def test_query_agent_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/agent/query"
        body = json.loads(request.content.decode())
        assert body["question"] == "Summarize leave policy"
        return httpx.Response(
            200,
            json={
                "question": "Summarize leave policy",
                "final_answer": "Leave policy summary.",
                "tool_executions": [],
                "planner_output": None,
                "planner_decision": None,
                "evaluation_context": None,
                "tool_validation": None,
                "correlation_id": "a-1",
                "trace_id": "t-1",
                "run_id": "r-1",
                "run_url": None,
                "latency_ms": 12.5,
                "metadata": {},
            },
        )

    with _client(handler) as client:
        result = client.query_agent("Summarize leave policy")
    assert result["final_answer"] == "Leave policy summary."
    assert result["trace_id"] == "t-1"


def test_api_failure_maps_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": "RAG retrieval failed"})

    with _client(handler) as client:
        with pytest.raises(ApiClientError) as exc:
            client.query_rag("hello")
    assert "RAG retrieval failed" in exc.value.message
    assert exc.value.status_code == 502


def test_timeout_raises_friendly_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = ApiClient(
        "http://test",
        timeout_seconds=1.0,
        client=httpx.Client(
            base_url="http://test",
            transport=httpx.MockTransport(handler),
        ),
    )
    with client:
        with pytest.raises(ApiClientError) as exc:
            client.get_health()
    assert "timed out" in exc.value.message.lower()


def test_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", headers={"content-type": "text/plain"})

    with _client(handler) as client:
        with pytest.raises(ApiClientError) as exc:
            client.get_health()
    assert "malformed" in exc.value.message.lower()


def test_empty_question_rejected_locally() -> None:
    with _client(lambda r: httpx.Response(200)) as client:
        with pytest.raises(ApiClientError) as exc:
            client.query_rag("   ")
        assert "empty" in exc.value.message.lower()
        with pytest.raises(ApiClientError):
            client.query_agent("")


def test_unavailable_backend() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with _client(handler) as client:
        with pytest.raises(ApiClientError) as exc:
            client.get_health()
    assert "unavailable" in exc.value.message.lower()


def test_run_demo_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/demo/run"
        return httpx.Response(
            200,
            json={
                "scenario_id": "agent-tools-foundation-001",
                "mode": "offline",
                "question": "q",
                "expected_tools": ["search_docs", "summarize"],
                "agent_result": {
                    "question": "q",
                    "final_answer": "a",
                    "tool_executions": [],
                    "correlation_id": "c",
                    "metadata": {},
                },
                "evaluation_report": {
                    "question": "q",
                    "answer": "a",
                    "metrics": [],
                    "overall_score": 0.9,
                    "passed": True,
                    "evaluation_time": "2026-01-01T00:00:00Z",
                    "latency": 1.0,
                    "pass_threshold": 0.7,
                },
                "quality_decision": {
                    "status": "PASS",
                    "reason": "ok",
                    "overall_score": 0.9,
                },
                "report_paths": {},
                "metadata": {},
            },
        )

    with _client(handler) as client:
        result = client.run_demo(live=False)
    assert result["scenario_id"] == "agent-tools-foundation-001"
    assert result["quality_decision"]["status"] == "PASS"
