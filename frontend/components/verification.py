"""Natural employee verification UX (no internal tool names exposed)."""

from __future__ import annotations

import re
from typing import Any

import streamlit as st

from frontend.api_client import ApiClient, ApiClientError
from frontend.components.chat import append_message, request_chat_scroll
from frontend.components.employee_workspace import request_workspace_sidebar_open
from frontend.user_messages import friendly_error, technical_detail

_EMPLOYEE_ID_RE = re.compile(r"^E-\d{4}$", re.IGNORECASE)
_DIGITS_ONLY_RE = re.compile(r"^\d+$")
# Bare digits are accepted only when they already match the #### portion of E-####.
_BARE_EMPLOYEE_DIGITS_RE = re.compile(r"^\d{4}$")

# Deterministic demo range used while / until the employee verify API is live.
_DEMO_EMPLOYEE_IDS = {f"E-{n}" for n in range(1101, 1131)}

_EMPLOYEE_INTENT_PHRASES = (
    "my leave",
    "leave balance",
    "vacation days",
    "vacation day",
    "days do i have",
    "days left",
    "leave history",
    "my benefits",
    "my benefit",
    "pending task",
    "pending action",
    "my pending",
    "my manager",
    "my salary",
    "my pto",
    "time off i have",
    "how many did i take",
    "carry them forward",
    "carry forward",
)

VERIFY_PROMPT = (
    "Before I access your employee information, I need to verify your "
    "employee ID.\n\n"
    "Please enter your employee ID, for example E-1101."
)

VERIFY_SUCCESS = "You're verified ✓"


def needs_employee_verification(question: str) -> bool:
    """Heuristic: personal HR / leave questions require a verified employee.

    Standalone numeric input (e.g. ``123``) never triggers the verification
    gate — only explicit ``E-####`` ids or personal HR phrasing do.
    """
    text = (question or "").strip()
    if not text:
        return False
    if _DIGITS_ONLY_RE.match(text):
        return False
    if looks_like_employee_id(text):
        return False
    lowered = text.lower()
    return any(phrase in lowered for phrase in _EMPLOYEE_INTENT_PHRASES)


def looks_like_employee_id(value: str) -> bool:
    """True only for explicit ``E-####`` (after light normalization)."""
    cleaned = (value or "").strip().upper().replace(" ", "")
    return bool(_EMPLOYEE_ID_RE.match(cleaned))


def normalize_employee_id(value: str) -> str:
    """Normalize spacing/case for employee IDs like ``E-1101``.

    Bare digits are prefixed with ``E-`` only when they are exactly four
    digits (the expected id body). Inputs like ``123`` stay ``123`` so they
    do not accidentally enter verification as ``E-123``.
    """
    cleaned = (value or "").strip().upper().replace(" ", "")
    if _BARE_EMPLOYEE_DIGITS_RE.match(cleaned):
        return f"E-{cleaned}"
    return cleaned


def verify_employee_id(client: ApiClient, employee_id: str) -> dict[str, Any]:
    """Verify via backend when available; fall back to demo ID range.

    Backend path (preferred): ``POST /api/v1/employees/verify``.
    Fallback keeps the interview demo usable before that route ships.
    """
    eid = normalize_employee_id(employee_id)
    if not _EMPLOYEE_ID_RE.match(eid):
        raise ApiClientError(_VERIFY_FORMAT_MSG, status_code=400)

    try:
        return client.verify_employee(eid)
    except ApiClientError as exc:
        message = (exc.message or "").lower()
        # Route not deployed yet, or API temporarily unreachable — demo IDs.
        if (
            exc.status_code == 404
            or "not found" in message
            or "unavailable" in message
        ):
            return _demo_verify(eid)
        raise


_VERIFY_FORMAT_MSG = (
    "That employee ID couldn't be verified.\n\n"
    "Employee IDs should follow the format E-1101.\n"
    "Please recheck and try again."
)


def _demo_verify(employee_id: str) -> dict[str, Any]:
    if employee_id in _DEMO_EMPLOYEE_IDS:
        return {
            "verified": True,
            "employee_id": employee_id,
            "source": "demo_fallback",
        }
    raise ApiClientError(_VERIFY_FORMAT_MSG, status_code=400)


def render_verification_form(client: ApiClient) -> None:
    """Show a simple Employee ID form when verification is required."""
    with st.container(border=True):
        st.markdown("**Verify your employee ID**")
        st.caption(VERIFY_PROMPT)
        with st.form("employee_verify_form", clear_on_submit=False):
            employee_id = st.text_input(
                "Employee ID",
                placeholder="E-1101",
                help="Your employee ID starts with E-",
            )
            st.caption("Your employee ID starts with E-")
            submitted = st.form_submit_button(
                "Verify",
                type="primary",
                icon=":material/verified_user:",
            )

        if st.session_state.get("verify_error"):
            st.error(st.session_state.verify_error, icon=":material/error:")

    if not submitted:
        return

    eid = normalize_employee_id(employee_id)
    try:
        result = verify_employee_id(client, eid)
    except Exception as exc:  # noqa: BLE001 — map any failure to friendly copy
        st.session_state.last_error_technical = technical_detail(exc)
        st.session_state.verify_error = friendly_error(exc)
        st.rerun()
        return

    verified_id = str(result.get("employee_id") or eid)
    st.session_state.verified_employee_id = verified_id
    st.session_state.awaiting_verification = False
    st.session_state.verify_error = None
    st.session_state.last_verify_result = result
    append_message("assistant", VERIFY_SUCCESS)
    request_chat_scroll(force=True)
    request_workspace_sidebar_open()
    st.toast(VERIFY_SUCCESS, icon=":material/check_circle:")

    pending = st.session_state.pop("pending_question", None)
    if pending:
        st.session_state.queued_prompt = pending
        st.session_state.resume_after_verify = True
    st.rerun()


def render_verified_indicator() -> None:
    """Small contextual badge for the verified employee."""
    eid = st.session_state.get("verified_employee_id")
    if not eid:
        return
    st.badge(
        f"Verified · {eid}",
        icon=":material/check_circle:",
        color="green",
    )
