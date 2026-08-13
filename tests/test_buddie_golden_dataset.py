"""Sprint 17 — deterministic validation for Buddie golden dataset foundation.

No LLM judge. Validates schema shape, ids, categories, tool allowlist, and
case-count bounds only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.mcp.schemas import EXPECTED_MCP_TOOL_NAMES

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "golden_dataset"
    / "buddie_golden_cases.json"
)

REQUIRED_CASE_FIELDS = (
    "id",
    "category",
    "user_query",
    "expected_answer",
    "expected_behavior",
)

ALLOWED_CATEGORIES = frozenset(
    {
        "leave_hr",
        "holidays",
        "benefits_policies",
        "rag_knowledge",
        "multi_tool",
        "negative_unknown",
    }
)

ALLOWED_BEHAVIORS = frozenset(
    {
        "answer_from_tool",
        "answer_from_rag",
        "combine_tools",
        "refuse_or_insufficient",
        "require_verification",
        "require_hitl_confirmation",
    }
)

# Tools that appear in Buddie employee / RAG / MCP surfaces (not a runtime registry).
ALLOWED_TOOLS = frozenset(
    {
        *EXPECTED_MCP_TOOL_NAMES,
        "get_upcoming_leave",
        "get_upcoming_holidays",
        "get_pending_actions",
        "get_payroll_summary",
        "get_attendance_summary",
        "verify_employee",
        "search_docs",
        "summarize",
        "calculator",
        "search",
    }
)

MIN_CASES = 20
MAX_CASES = 30


def _load_dataset() -> dict[str, Any]:
    raw = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


@pytest.fixture(scope="module")
def dataset() -> dict[str, Any]:
    return _load_dataset()


@pytest.fixture(scope="module")
def cases(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    loaded = dataset.get("cases")
    assert isinstance(loaded, list)
    return loaded


def test_golden_dataset_file_exists() -> None:
    assert GOLDEN_PATH.is_file()


def test_golden_dataset_loads_as_object(dataset: dict[str, Any]) -> None:
    assert dataset.get("version")
    assert dataset.get("name") == "buddie_golden_cases"
    assert isinstance(dataset.get("default_session"), dict)
    assert dataset["default_session"].get("verified_employee_id") == "E-1101"
    assert isinstance(dataset.get("cases"), list)


def test_case_count_within_sprint_bounds(cases: list[dict[str, Any]]) -> None:
    assert MIN_CASES <= len(cases) <= MAX_CASES


def test_case_ids_are_unique_and_non_blank(cases: list[dict[str, Any]]) -> None:
    ids = [case["id"] for case in cases]
    assert all(isinstance(i, str) and i.strip() for i in ids)
    assert len(ids) == len(set(ids))


def test_required_fields_present_and_non_blank(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        for field in REQUIRED_CASE_FIELDS:
            assert field in case, f"{case.get('id')}: missing {field}"
            value = case[field]
            assert isinstance(value, str) and value.strip(), (
                f"{case['id']}: blank required field {field}"
            )


def test_categories_are_allowed(cases: list[dict[str, Any]]) -> None:
    categories = {case["category"] for case in cases}
    assert categories <= ALLOWED_CATEGORIES
    # Sprint coverage: every major bucket should appear at least once.
    assert ALLOWED_CATEGORIES <= categories


def test_expected_behaviors_are_allowed(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        assert case["expected_behavior"] in ALLOWED_BEHAVIORS, case["id"]


def test_optional_context_and_tools_shapes(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        cid = case["id"]
        if "expected_context" in case:
            ctx = case["expected_context"]
            assert isinstance(ctx, list), cid
            assert all(isinstance(item, str) and item.strip() for item in ctx), cid

        expected_tool = case.get("expected_tool")
        expected_tools = case.get("expected_tools")

        if expected_tool is not None:
            assert isinstance(expected_tool, str) and expected_tool.strip(), cid
            assert expected_tool in ALLOWED_TOOLS, f"{cid}: unknown tool {expected_tool}"

        if expected_tools is not None:
            assert isinstance(expected_tools, list), cid
            assert all(isinstance(t, str) and t.strip() for t in expected_tools), cid
            unknown = set(expected_tools) - ALLOWED_TOOLS
            assert not unknown, f"{cid}: unknown tools {unknown}"

        if expected_tool and expected_tools:
            assert expected_tool in expected_tools, (
                f"{cid}: expected_tool not in expected_tools"
            )

        if expected_tool and expected_tools is None:
            # Single-tool cases may omit the list; that is valid.
            pass


def test_no_unexpected_top_level_case_keys(cases: list[dict[str, Any]]) -> None:
    allowed = {
        "id",
        "category",
        "user_query",
        "expected_answer",
        "expected_context",
        "expected_tool",
        "expected_tools",
        "expected_behavior",
        "evaluation_notes",
    }
    for case in cases:
        extra = set(case) - allowed
        assert not extra, f"{case['id']}: unexpected keys {extra}"


def test_marked_uncertain_answers_are_explicit(cases: list[dict[str, Any]]) -> None:
    """Cases without a safely fixed answer must use an explicit marker."""
    markers = ("[ANSWER_NOT_FULLY_SPECIFIED]", "[ANSWER_NOT_IN_CORPUS]")
    uncertain = [
        c
        for c in cases
        if any(m in c["expected_answer"] for m in markers)
    ]
    assert uncertain, "expected at least one explicitly marked uncertain answer"
    for case in uncertain:
        assert case.get("evaluation_notes"), case["id"]


def test_hitl_case_excludes_write_tool_on_draft(cases: list[dict[str, Any]]) -> None:
    hitl = [c for c in cases if c["expected_behavior"] == "require_hitl_confirmation"]
    assert hitl
    for case in hitl:
        tools = case.get("expected_tools") or []
        assert "create_leave_request" not in tools, case["id"]
