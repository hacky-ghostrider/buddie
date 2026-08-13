"""Tests for the authenticated Employee Workspace sidebar."""

from __future__ import annotations

from frontend.components.employee_workspace import (
    OPTIONAL_ACTIONS,
    QUICK_ACCESS_ACTIONS,
    WorkspaceAction,
    action_for_label,
    employee_workspace_visible,
    is_employee_verified,
    queue_workspace_action,
    verified_employee_id,
)
from frontend.components.response_formatting import (
    format_assistant_answer,
    format_leave_balance,
    format_leave_history,
    format_upcoming_holidays,
    looks_like_raw_structured_dump,
)
from frontend.components.verification import needs_employee_verification


class _FakeSession(dict):
    """Minimal stand-in for ``st.session_state`` (attr + mapping access)."""

    def __getattr__(self, key: str):  # noqa: ANN001
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value) -> None:  # noqa: ANN001
        self[key] = value

    def get(self, key, default=None):  # noqa: ANN001
        return dict.get(self, key, default)

    def pop(self, key, default=None):  # noqa: ANN001
        return dict.pop(self, key, default)


def test_workspace_hidden_before_verification(monkeypatch) -> None:
    state = _FakeSession(verified_employee_id=None)
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    assert is_employee_verified() is False
    assert employee_workspace_visible() is False
    assert verified_employee_id() is None


def test_workspace_visible_after_verification(monkeypatch) -> None:
    state = _FakeSession(verified_employee_id="E-1102")
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    assert is_employee_verified() is True
    assert employee_workspace_visible() is True
    assert verified_employee_id() == "E-1102"


def test_sidebar_expands_only_when_verified(monkeypatch) -> None:
    from frontend.components.employee_workspace import sidebar_state_for_session

    unverified = _FakeSession(verified_employee_id=None)
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        unverified,
    )
    assert sidebar_state_for_session() == "collapsed"

    verified = _FakeSession(verified_employee_id="E-1101")
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        verified,
    )
    assert sidebar_state_for_session() == "expanded"


def test_ensure_workspace_sidebar_open_emits_expand_script(monkeypatch) -> None:
    from frontend.components import employee_workspace as ws

    state = _FakeSession(
        verified_employee_id="E-1101",
        _buddie_sidebar_open_nonce=0,
        _buddie_open_sidebar=True,
        _buddie_sidebar_opened_once=False,
    )
    monkeypatch.setattr(ws.st, "session_state", state)
    captured: dict[str, object] = {}

    def _fake_html(html: str, *, height: int = 0, width: int = 0) -> None:
        captured["html"] = html
        captured["height"] = height

    monkeypatch.setattr(ws.components, "html", _fake_html)
    ws.ensure_workspace_sidebar_open()
    html = str(captured["html"])
    assert 'data-testid="stExpandSidebarButton"' in html
    assert "shouldOpen = true" in html
    assert "click()" in html
    assert state["_buddie_sidebar_open_nonce"] == 1
    assert state["_buddie_sidebar_opened_once"] is True
    # Force flag is consumed so later collapses are not yanked open.
    assert state.get("_buddie_open_sidebar") in {False, None}


def test_request_workspace_sidebar_open_sets_flag(monkeypatch) -> None:
    from frontend.components import employee_workspace as ws

    state = _FakeSession()
    monkeypatch.setattr(ws.st, "session_state", state)
    ws.request_workspace_sidebar_open()
    assert state["_buddie_open_sidebar"] is True


def test_ensure_does_not_force_open_after_first_open(monkeypatch) -> None:
    from frontend.components import employee_workspace as ws

    state = _FakeSession(
        verified_employee_id="E-1101",
        _buddie_sidebar_open_nonce=2,
        _buddie_sidebar_opened_once=True,
    )
    monkeypatch.setattr(ws.st, "session_state", state)
    captured: dict[str, object] = {}

    def _fake_html(html: str, *, height: int = 0, width: int = 0) -> None:
        captured["html"] = html

    monkeypatch.setattr(ws.components, "html", _fake_html)
    ws.ensure_workspace_sidebar_open()
    assert "shouldOpen = false" in str(captured["html"])
    # Still restyles expand control for reopen-after-collapse.
    assert 'stExpandSidebarButton' in str(captured["html"])


def test_verified_employee_id_comes_from_session(monkeypatch) -> None:
    state = _FakeSession(verified_employee_id="E-1115")
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    assert verified_employee_id() == "E-1115"

    state["verified_employee_id"] = "E-1107"
    assert verified_employee_id() == "E-1107"


def test_quick_access_maps_to_existing_capabilities() -> None:
    by_label = {action.label: action for action in QUICK_ACCESS_ACTIONS}
    assert by_label["My Leave"].tool_name == "get_leave_balance"
    assert by_label["Leave History"].tool_name == "get_leave_history"
    assert by_label["Upcoming Holidays"].tool_name == "get_upcoming_holidays"
    assert by_label["Pending Actions"].tool_name == "get_pending_actions"
    assert action_for_label("Employee Profile") is not None
    assert action_for_label("Employee Profile").tool_name == "get_employee_profile"
    assert len(OPTIONAL_ACTIONS) == 1


def test_my_leave_queues_leave_balance_capability(monkeypatch) -> None:
    state = _FakeSession(verified_employee_id="E-1101")
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    action = action_for_label("My Leave")
    assert action is not None
    queue_workspace_action(action)
    assert state["queued_prompt"] == action.question
    assert needs_employee_verification(action.question) is True


def test_leave_history_queues_leave_history_capability(monkeypatch) -> None:
    state = _FakeSession(verified_employee_id="E-1101")
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    action = action_for_label("Leave History")
    assert action is not None
    queue_workspace_action(action)
    assert state["queued_prompt"] == action.question
    assert "leave history" in action.question.lower()
    assert action.tool_name == "get_leave_history"


def test_protected_sidebar_action_cannot_bypass_verification(monkeypatch) -> None:
    state = _FakeSession(
        verified_employee_id=None,
        awaiting_verification=False,
        pending_question=None,
        queued_prompt=None,
    )
    monkeypatch.setattr(
        "frontend.components.employee_workspace.st.session_state",
        state,
    )
    action = action_for_label("My Leave")
    assert action is not None
    queue_workspace_action(action)
    assert state.get("queued_prompt") is None
    assert state["awaiting_verification"] is True
    assert state["pending_question"] == action.question


def test_holidays_do_not_fabricate_data() -> None:
    formatted = format_upcoming_holidays(
        {
            "upcoming_holidays": [],
            "next_holiday": None,
        }
    )
    assert "No upcoming company holidays found." in formatted
    assert "New Year" not in formatted
    assert "{" not in formatted


def test_pending_actions_empty_is_not_fabricated() -> None:
    answer = format_assistant_answer(
        "AGENT",
        {
            "final_answer": "You have no pending HR actions.",
            "tool_executions": [
                {
                    "tool_name": "get_pending_actions",
                    "status": "success",
                    "output": {
                        "employee_id": "E-1101",
                        "pending_actions": [],
                        "count": 0,
                    },
                }
            ],
        },
    )
    assert "no pending" in answer.lower()
    assert "fabricat" not in answer.lower()
    assert answer.strip() != "{}"


def test_leave_balance_format_is_clean_not_raw_dict() -> None:
    formatted = format_leave_balance(
        {
            "employee_id": "E-1101",
            "leave_balance": {"vacation": 14, "sick": 8, "personal": 3},
        }
    )
    assert "Leave Balance" in formatted
    assert "Vacation" in formatted
    assert "14 days" in formatted
    assert "sick=8" not in formatted
    assert not formatted.strip().startswith("{")
    assert not looks_like_raw_structured_dump(formatted)


def test_leave_history_format_is_table_not_raw_dict() -> None:
    formatted = format_leave_history(
        {
            "employee_id": "E-1101",
            "leave_history": [
                {
                    "date": "2024-01-03",
                    "type": "VACATION",
                    "days": 1,
                    "status": "APPROVED",
                }
            ],
            "total_days": 1,
        }
    )
    assert "Leave History" in formatted
    assert "Jan 03 2024" in formatted
    assert "leave_history" not in formatted
    assert not looks_like_raw_structured_dump(formatted)


def test_format_assistant_answer_prefers_structured_leave_balance() -> None:
    answer = format_assistant_answer(
        "AGENT",
        {
            "final_answer": (
                "Leave balance for E-1101: vacation=14 days, "
                "sick=8 days, personal=3 days."
            ),
            "tool_executions": [
                {
                    "tool_name": "get_leave_balance",
                    "status": "success",
                    "output": {
                        "employee_id": "E-1101",
                        "leave_balance": {
                            "vacation": 14,
                            "sick": 8,
                            "personal": 3,
                        },
                    },
                }
            ],
        },
    )
    assert "Leave Balance" in answer
    assert "14 days" in answer
    assert "{" not in answer


def test_workspace_actions_are_not_decorative() -> None:
    for action in (*QUICK_ACCESS_ACTIONS, *OPTIONAL_ACTIONS):
        assert isinstance(action, WorkspaceAction)
        assert action.question.strip()
        assert action.tool_name.strip()
        assert action.label.strip()


def test_chat_and_developer_imports_remain_available() -> None:
    # Smoke: existing chat / developer modules still import cleanly with workspace.
    from frontend.components import chat, developer_panel, employee_workspace

    assert hasattr(chat, "init_chat_state")
    assert hasattr(developer_panel, "render_developer_menu")
    assert hasattr(employee_workspace, "render_employee_workspace")
