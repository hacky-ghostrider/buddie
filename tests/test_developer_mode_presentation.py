"""Developer Mode metadata extraction and response presentation helpers."""

from __future__ import annotations

from app.agent.router import _format_employee_output, _format_leave_history
from app.employees.service import normalize_employee_id
from frontend.components.execution_metadata import (
    extract_execution_metadata,
    sanitize_for_developer_view,
)
from frontend.components.response_formatting import (
    format_assistant_answer,
    format_leave_history,
    looks_like_raw_structured_dump,
)


def test_format_leave_history_readable() -> None:
    formatted = _format_leave_history(
        {
            "employee_id": "E-1101",
            "year": 2025,
            "total_days": 5,
            "leave_history": [
                {
                    "date": "2025-03-10",
                    "type": "VACATION",
                    "days": 3,
                    "status": "APPROVED",
                    "reason": "Family trip",
                },
                {
                    "date": "2025-06-01",
                    "type": "SICK",
                    "days": 2.0,
                    "status": "APPROVED",
                    "reason": "Flu",
                },
            ],
        }
    )
    assert "Leave History — E-1101" in formatted
    assert "Mar 10 2025" in formatted
    assert "Vacation" in formatted
    assert "Approved" in formatted
    assert "Total leave used: 5 days" in formatted
    assert "leave_history" not in formatted
    assert not formatted.strip().startswith("{")


def test_presentation_format_leave_history_table() -> None:
    formatted = format_leave_history(
        {
            "employee_id": "E-1101",
            "total_days": 47.0,
            "leave_history": [
                {
                    "date": "2024-01-03",
                    "type": "VACATION",
                    "days": 1,
                    "status": "APPROVED",
                },
                {
                    "date": "2024-01-11",
                    "type": "SICK",
                    "days": 1,
                    "status": "APPROVED",
                },
            ],
        }
    )
    assert "Leave History — E-1101" in formatted
    assert "Jan 03 2024" in formatted
    assert "Total leave used: 47 days" in formatted
    assert "{" not in formatted.replace("```", "")


def test_format_employee_output_leave_history() -> None:
    text = _format_employee_output(
        "get_leave_history",
        {
            "employee_id": "E-1101",
            "leave_history": [],
            "total_days": 0,
        },
    )
    assert text is not None
    assert "No leave history" in text


def test_format_pending_actions_list() -> None:
    text = _format_employee_output(
        "get_pending_actions",
        {
            "employee_id": "E-1101",
            "count": 1,
            "pending_actions": [
                {
                    "action_id": "A-1",
                    "description": "Submit benefits enrollment",
                    "due_date": "2026-09-01",
                    "status": "PENDING",
                }
            ],
        },
    )
    assert text is not None
    assert "Submit benefits enrollment" in text
    assert "2026-09-01" in text


def test_raw_dict_dump_is_detected_and_reformatted() -> None:
    raw = str(
        {
            "employee_id": "E-1101",
            "year": None,
            "leave_history": [
                {
                    "date": "2024-01-03",
                    "type": "VACATION",
                    "days": 1,
                    "status": "APPROVED",
                }
            ],
            "total_days": 1.0,
        }
    )
    assert looks_like_raw_structured_dump(raw)
    answer = format_assistant_answer(
        "AGENT",
        {
            "final_answer": raw,
            "tool_executions": [
                {
                    "tool_name": "get_leave_history",
                    "status": "success",
                    "output": {
                        "employee_id": "E-1101",
                        "leave_history": [
                            {
                                "date": "2024-01-03",
                                "type": "VACATION",
                                "days": 1,
                                "status": "APPROVED",
                            }
                        ],
                        "total_days": 1.0,
                    },
                }
            ],
        },
    )
    assert "Leave History — E-1101" in answer
    assert "leave_history" not in answer


def test_sanitize_for_developer_view_redacts_secrets() -> None:
    cleaned = sanitize_for_developer_view(
        {
            "intent_route": "employee",
            "api_key": "sk-live-secret",
            "openai_api_key": "sk-other",
            "nested": {"token": "abc", "ok": 1},
        }
    )
    assert cleaned["intent_route"] == "employee"
    assert cleaned["api_key"] == "[redacted]"
    assert cleaned["openai_api_key"] == "[redacted]"
    assert cleaned["nested"]["token"] == "[redacted]"
    assert cleaned["nested"]["ok"] == 1


def test_extract_execution_metadata_from_agent_payload() -> None:
    payload = {
        "question": "Show my leave history",
        "final_answer": "Leave History — E-1101",
        "correlation_id": "corr-1",
        "trace_id": "trace-1",
        "run_url": "https://smith.langchain.com/public/demo",
        "latency_ms": 12.5,
        "tool_executions": [
            {
                "tool_name": "get_leave_history",
                "status": "success",
                "arguments": {"year": 2025},
                "output": {"leave_history": [{"date": "2025-01-01"}], "total_days": 1},
                "latency_ms": 3.0,
            }
        ],
        "evaluation_context": {
            "retrieved_documents": [
                {
                    "id": "d1",
                    "metadata": {"source": "employee_handbook.pdf"},
                    "score": 0.91,
                }
            ],
            "retrieved_chunks": [],
            "model": "mock-llm",
        },
        "metadata": {
            "normalized_input": "Show my leave history",
            "detected_intent": "employee",
            "selected_route": "employee",
            "verification_status": "verified",
            "verified_employee_id": "E-1101",
            "rag_used": False,
            "tools_invoked": [
                {
                    "tool_name": "get_leave_history",
                    "status": "success",
                    "arguments": {"year": 2025},
                    "result_summary": "leave_history entries=1 total_days=1",
                    "latency_ms": 3.0,
                    "error": None,
                }
            ],
            "api_key": "should-be-redacted",
        },
        "planner_output": {"intent_route": "employee"},
    }
    snapshot = extract_execution_metadata(payload, "AGENT")
    assert snapshot["original_input"] == "Show my leave history"
    assert snapshot["normalized_input"] == "Show my leave history"
    assert snapshot["detected_intent"] == "employee"
    assert snapshot["selected_route"] == "employee"
    assert snapshot["verification_status"] == "verified"
    assert snapshot["verification_required"] is True
    assert snapshot["tools_invoked"][0]["tool_name"] == "get_leave_history"
    assert snapshot["langsmith_run_url"].startswith("https://")
    assert snapshot["latency_ms"] == 12.5
    assert snapshot["model"] == "mock-llm"
    assert "should-be-redacted" not in str(snapshot)
    assert snapshot["retrieved_sources"] == ["employee_handbook.pdf"]
    assert snapshot["retrieval_scores"] == [0.91]


def test_extract_execution_metadata_knowledge_rag() -> None:
    payload = {
        "question": "What is the vacation policy?",
        "answer": "Paid leave accrues per the handbook.",
        "retrieved_documents": [
            {
                "id": "c1",
                "metadata": {"source": "Employee Handbook — Leave Policy"},
                "score": 0.88,
            }
        ],
        "retrieval_metadata": {"retrieved_count": 1, "top_k": 4},
        "generation_metadata": {"model": "mock-llm"},
        "latency": {"total_ms": 9.5},
    }
    snapshot = extract_execution_metadata(payload, "RAG")
    assert snapshot["rag_used"] is True
    assert snapshot["detected_intent"] == "knowledge"
    assert snapshot["selected_route"] == "rag"
    assert snapshot["verification_required"] is False
    assert snapshot["verification_status"] == "not_required"
    assert snapshot["tools_invoked"] == []
    assert "Employee Handbook — Leave Policy" in snapshot["retrieved_sources"]
    assert snapshot["top_k"] == 4
    assert snapshot["retrieval_scores"] == [0.88]
    assert snapshot["latency_ms"] == 9.5
    assert snapshot["model"] == "mock-llm"


def test_extract_execution_metadata_no_fake_fields() -> None:
    snapshot = extract_execution_metadata(
        {
            "question": "hello",
            "final_answer": "Hi!",
            "metadata": {
                "normalized_input": "hello",
                "detected_intent": "conversation",
                "selected_route": "conversation",
                "rag_used": False,
                "tools_invoked": [],
            },
        },
        "AGENT",
    )
    assert snapshot["rag_used"] is False
    assert snapshot["tools_invoked"] == []
    assert snapshot["langsmith_run_url"] is None
    assert "api_key" not in snapshot
    assert snapshot.get("latency_ms") is None or isinstance(
        snapshot.get("latency_ms"), (int, float)
    )


def test_backend_normalize_employee_id_rejects_short_digits() -> None:
    assert normalize_employee_id("1101") == "E-1101"
    assert normalize_employee_id("123") == "123"
    assert normalize_employee_id("E-1101") == "E-1101"
