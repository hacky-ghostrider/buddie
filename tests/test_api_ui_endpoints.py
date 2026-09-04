"""API smoke tests for agent / demo / reports thin endpoints (mocked services)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agent.models import AgentRunResult
from app.api.deps import get_agent_service
from app.config.settings import Settings, get_settings
from app.demo.models import DemoResult
from app.evaluation.quality.decision import QualityDecision, QualityStatus
from app.evaluation.report import EvaluationReport
from app.main import create_app
from app.retrieval.models import RetrievedDocument

pytestmark = pytest.mark.smoke


def _client_with_agent(mock_agent: MagicMock) -> TestClient:
    get_settings.cache_clear()
    settings = Settings(app_name="rag-evaluation-platform", app_env="test")
    application = create_app(settings=settings)
    application.dependency_overrides[get_agent_service] = lambda: mock_agent
    return TestClient(application)


def test_agent_query_endpoint_delegates() -> None:
    mock_agent = MagicMock()
    mock_agent.run.return_value = AgentRunResult(
        question="2 + 2",
        final_answer="4",
        correlation_id="corr-1",
        latency_ms=5.0,
    )
    client = _client_with_agent(mock_agent)
    response = client.post("/api/v1/agent/query", json={"question": "2 + 2"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["final_answer"] == "4"
    mock_agent.run.assert_called_once()


def test_agent_query_rejects_blank_question() -> None:
    mock_agent = MagicMock()
    client = _client_with_agent(mock_agent)
    response = client.post("/api/v1/agent/query", json={"question": "  "})
    assert response.status_code == 422
    mock_agent.run.assert_not_called()


def test_reports_list_endpoint() -> None:
    get_settings.cache_clear()
    settings = Settings(app_name="rag-evaluation-platform", app_env="test")
    client = TestClient(create_app(settings=settings))
    response = client.get("/api/v1/reports")
    assert response.status_code == 200
    assert "files" in response.json()


def test_demo_endpoint_uses_runner(monkeypatch) -> None:
    get_settings.cache_clear()
    settings = Settings(app_name="rag-evaluation-platform", app_env="test")
    application = create_app(settings=settings)

    fake = DemoResult(
        scenario_id="agent-tools-foundation-001",
        mode="offline",
        question="Summarize the leave policy from the employee handbook.",
        expected_tools=["search_docs", "summarize"],
        agent_result=AgentRunResult(
            question="Summarize the leave policy from the employee handbook.",
            final_answer="Leave policy summary.",
            correlation_id="demo-1",
        ),
        evaluation_report=EvaluationReport.build(
            question="Summarize the leave policy from the employee handbook.",
            answer="Leave policy summary.",
            retrieved_documents=[
                RetrievedDocument(
                    id="d1",
                    text="Employees accrue paid leave.",
                    metadata={"source": "employee_handbook.pdf"},
                    score=0.9,
                )
            ],
            metrics=[],
            latency_ms=1.0,
            pass_threshold=0.0,
        ),
        quality_decision=QualityDecision(
            status=QualityStatus.PASS,
            reason="All gates passed",
            overall_score=0.9,
        ),
        report_paths={"evaluation_json": "data/demo/reports/evaluation.json"},
    )

    monkeypatch.setattr("app.api.v1.demo.run_canonical_demo", lambda **kwargs: fake)
    client = TestClient(application)
    response = client.post("/api/v1/demo/run", json={"live": False})
    assert response.status_code == 200
    body = response.json()
    assert body["scenario_id"] == "agent-tools-foundation-001"
    assert body["quality_decision"]["status"] == "PASS"
    assert body["expected_tools"] == ["search_docs", "summarize"]
