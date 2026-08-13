"""Buddie — Streamlit presentation layer over FastAPI.

Architecture:
    Browser → Streamlit → FastAPI → RAGService / AgentService / Demo runner
                                    → Evaluation / DeepEval / LangSmith / Gates

No RAG, agent, retrieval, or evaluation logic lives here. This module is the
employee-facing product shell; technical observability lives only in
Developer / Evaluation mode.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Allow `uv run streamlit run frontend/app.py` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from frontend.api_client import ApiClient, ApiClientError
from frontend.components.chat import (
    append_message,
    clear_conversation,
    init_chat_state,
    render_chat_autoscroll,
    render_chat_history,
    request_chat_scroll,
    store_response,
)
from frontend.components.developer_panel import (
    render_developer_console,
    render_developer_menu,
)
from frontend.components.employee_workspace import (
    render_employee_workspace,
    request_workspace_sidebar_open,
)
from frontend.components.quick_actions import render_quick_actions
from frontend.components.response_formatting import format_assistant_answer
from frontend.components.verification import (
    VERIFY_PROMPT,
    needs_employee_verification,
    render_verification_form,
    render_verified_indicator,
)
from frontend.config import load_config
from frontend.user_messages import friendly_error, technical_detail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend")

config = load_config()


def _sidebar_state_for_session() -> str:
    """Expand employee workspace after verify; keep it collapsed before.

    Kept local to ``app.py`` so Streamlit hot-reload cannot fail on a stale
    ``employee_workspace`` import while ``set_page_config`` runs at module load.
    """
    eid = st.session_state.get("verified_employee_id")
    return "expanded" if eid and str(eid).strip() else "collapsed"


# Collapse until verified; expand once the employee workspace has content.
# Reading session state here is intentional so verify/sign-out reruns flip
# Streamlit's sidebar chrome (content is still gated in render_employee_workspace).
st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout="wide",
    initial_sidebar_state=_sidebar_state_for_session(),
)

# Ocean Lagoon design tokens — keep in sync with .streamlit/config.toml
# Identity rule: ~70–80% neutral surfaces, ~20–30% lagoon/aqua accents.
st.html(
    """
    <style>
    :root {
        --buddie-bg: #F4FBFA;
        --buddie-surface: #FFFFFF;
        --buddie-primary: #0F766E;
        --buddie-secondary: #14B8A6;
        --buddie-accent: #22D3EE;
        --buddie-text: #16323A;
        --buddie-muted: #64748B;
        --buddie-border: #D9E7E5;
        --buddie-tint: #E6F7F5;
        --buddie-tint-strong: #D1FAF5;
        --buddie-focus-ring: rgba(15, 118, 110, 0.22);
    }
    .stAppDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }
    /* Keep Streamlit's top chrome from covering the brand row */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        /* Must stay visible so the sidebar expander remains clickable */
        overflow: visible !important;
    }
    /* Header is height:0 for brand layout; keep the sidebar opener usable.
       Critical after manual collapse — without this the expand control is clipped
       and the workspace cannot be reopened in Cursor/Chrome. */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] > div,
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        overflow: visible !important;
    }
    [data-testid="stExpandSidebarButton"],
    button[data-testid="stExpandSidebarButton"],
    button[kind="headerNoPadding"] {
        position: fixed !important;
        top: 0.7rem !important;
        left: 0.7rem !important;
        z-index: 1000000 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        width: 2.25rem !important;
        height: 2.25rem !important;
        min-width: 2.25rem !important;
        min-height: 2.25rem !important;
        background: var(--buddie-surface) !important;
        border: 1px solid var(--buddie-border) !important;
        border-radius: 0.65rem !important;
        box-shadow: 0 1px 2px rgba(22, 50, 58, 0.08) !important;
        color: var(--buddie-primary) !important;
    }
    .stApp {
        background: var(--buddie-bg);
    }
    .block-container {
        max-width: min(52rem, 100%);
        padding-top: 1.75rem !important;
        padding-bottom: 6.75rem !important;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        overflow: visible !important;
    }
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="stHtml"],
    [data-testid="stElementContainer"],
    [data-testid="stMainBlockContainer"] {
        overflow: visible !important;
    }

    /* Compact top wordmark — full Buddie identity lives in the welcome hero */
    .buddie-header-accent {
        display: block;
        height: 3px;
        width: 100%;
        margin: 0 0 0.85rem 0;
        border-radius: 999px;
        background: linear-gradient(
            90deg,
            var(--buddie-primary) 0%,
            var(--buddie-secondary) 55%,
            var(--buddie-accent) 100%
        );
    }
    .buddie-brand {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
        min-width: 9.5rem;
        padding-top: 0.15rem;
        overflow: visible;
    }
    .buddie-brand-title {
        margin: 0 !important;
        padding: 0.1rem 0 0 0 !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: var(--buddie-primary) !important;
        letter-spacing: -0.02em;
        line-height: 1.35 !important;
        white-space: nowrap;
    }
    .buddie-brand-mark {
        display: inline-block;
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--buddie-secondary), var(--buddie-accent));
        margin-right: 0.35rem;
        vertical-align: 0.05rem;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18);
    }
    /* Subtitle under the compact header wordmark */
    div[data-testid="stCaptionContainer"] p {
        color: var(--buddie-muted) !important;
        font-size: 0.85rem !important;
    }

    /* Welcome hierarchy: Buddie identity, then dark-teal prompt */
    .buddie-welcome-brand {
        margin: 0;
        color: var(--buddie-primary) !important;
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .buddie-welcome-sub {
        margin: 0.1rem 0 0 0;
        color: var(--buddie-muted) !important;
        font-size: 0.92rem;
        font-weight: 500;
    }
    .buddie-welcome h3,
    .buddie-welcome [data-testid="stMarkdownContainer"] h3 {
        color: var(--buddie-text) !important;
        font-weight: 650;
    }
    .buddie-welcome > p:not(.buddie-welcome-brand):not(.buddie-welcome-sub) {
        color: var(--buddie-muted) !important;
    }

    /* ---- Suggestion chips (st.pills → ButtonGroup) ---- */
    div[data-testid="stButtonGroup"] {
        flex-wrap: wrap !important;
        justify-content: center !important;
        max-width: 100%;
        row-gap: 0.45rem !important;
    }
    div[data-testid="stButtonGroup"] button {
        background: var(--buddie-surface) !important;
        border: 1px solid var(--buddie-border) !important;
        color: var(--buddie-text) !important;
        box-shadow: none !important;
    }
    div[data-testid="stButtonGroup"] button:hover {
        background: var(--buddie-tint) !important;
        border-color: var(--buddie-secondary) !important;
        color: var(--buddie-primary) !important;
    }
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] button[aria-pressed="true"],
    div[data-testid="stButtonGroup"] button[kind="primary"],
    div[data-testid="stButtonGroup"] button[data-testid="baseButton-primary"],
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-primary"] {
        background: var(--buddie-primary) !important;
        border-color: var(--buddie-primary) !important;
        color: #ffffff !important;
    }
    div[data-testid="stButtonGroup"] button[aria-checked="true"]:hover,
    div[data-testid="stButtonGroup"] button[kind="primary"]:hover,
    div[data-testid="stButtonGroup"] button[data-testid="baseButton-primary"]:hover,
    div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-primary"]:hover {
        background: var(--buddie-secondary) !important;
        border-color: var(--buddie-secondary) !important;
        color: #ffffff !important;
    }

    /* ---- Primary actions / More topics accents ---- */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: var(--buddie-primary) !important;
        border-color: var(--buddie-primary) !important;
        color: #ffffff !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: var(--buddie-secondary) !important;
        border-color: var(--buddie-secondary) !important;
    }
    button[kind="tertiary"]:hover,
    button[data-testid="baseButton-tertiary"]:hover,
    button[data-testid="stBaseButton-tertiary"]:hover {
        color: var(--buddie-primary) !important;
        background: var(--buddie-tint) !important;
    }

    /* ---- Chat messages ---- */
    div[data-testid="stChatMessage"] {
        background: var(--buddie-surface);
        border: 1px solid var(--buddie-border);
        border-radius: 12px;
    }

    /* ---- Chat input + send (Buddie primary action) ---- */
    div[data-testid="stBottom"] {
        background: linear-gradient(
            to top,
            var(--buddie-bg) 55%,
            rgba(244, 251, 250, 0.92) 78%,
            rgba(244, 251, 250, 0)
        ) !important;
        padding-bottom: 0.35rem !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        max-width: min(52rem, 100%) !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    div[data-testid="stChatInput"] {
        background: transparent !important;
        max-width: 100%;
    }
    div[data-testid="stChatInput"] > div {
        background: var(--buddie-surface) !important;
        border: 1.5px solid var(--buddie-border) !important;
        border-radius: 1.25rem !important;
        box-shadow: 0 1px 2px rgba(22, 50, 58, 0.04) !important;
    }
    div[data-testid="stChatInput"]:focus-within > div {
        border-color: var(--buddie-primary) !important;
        box-shadow: 0 0 0 3px var(--buddie-focus-ring) !important;
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInputTextArea"] {
        font-size: 1.05rem;
        color: var(--buddie-text) !important;
        background: transparent !important;
    }
    button[data-testid="stChatInputSubmitButton"] {
        background-color: var(--buddie-primary) !important;
        border: 1px solid var(--buddie-primary) !important;
        color: #ffffff !important;
        border-radius: 999px !important;
        opacity: 1 !important;
    }
    button[data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
        background-color: var(--buddie-secondary) !important;
        border-color: var(--buddie-secondary) !important;
    }
    button[data-testid="stChatInputSubmitButton"]:disabled {
        background-color: var(--buddie-primary) !important;
        border-color: var(--buddie-primary) !important;
        color: #ffffff !important;
        opacity: 0.45 !important;
    }
    button[data-testid="stChatInputSubmitButton"] svg,
    button[data-testid="stChatInputSubmitButton"] span {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* ---- Developer Mode ---- */
    .buddie-dev-shell {
        border-left: 3px solid var(--buddie-primary);
        padding-left: 0.75rem;
        margin: 0.25rem 0 0.75rem 0;
    }
    .buddie-dev-title {
        margin: 0;
        color: var(--buddie-primary);
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .buddie-dev-caption {
        margin: 0.2rem 0 0 0;
        color: var(--buddie-muted);
        font-size: 0.88rem;
    }
    .buddie-dev-dot {
        display: inline-block;
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 999px;
        background: var(--buddie-accent);
        margin-right: 0.4rem;
        vertical-align: middle;
    }
    span[data-testid="stBadge"] {
        border-color: transparent !important;
    }

    /* ---- Employee workspace sidebar (verified sessions only) ---- */
    section[data-testid="stSidebar"] {
        background: var(--buddie-bg) !important;
        border-right: 1px solid var(--buddie-border) !important;
    }
    section[data-testid="stSidebar"] > div {
        background: var(--buddie-bg) !important;
    }
    .buddie-ws-brand {
        margin: 0.15rem 0 0.1rem 0 !important;
        padding: 0 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: var(--buddie-primary) !important;
        letter-spacing: -0.02em;
        line-height: 1.3 !important;
    }
    .buddie-ws-section {
        margin: 0.95rem 0 0.35rem 0 !important;
        color: var(--buddie-muted) !important;
        font-size: 0.72rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .buddie-ws-identity {
        background: var(--buddie-surface);
        border: 1px solid var(--buddie-border);
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
    }
    .buddie-ws-eid {
        color: var(--buddie-text);
        font-size: 1.05rem;
        font-weight: 650;
        letter-spacing: -0.01em;
    }
    .buddie-ws-verified {
        margin-top: 0.15rem;
        color: var(--buddie-primary);
        font-size: 0.85rem;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] button[kind="tertiary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"] {
        justify-content: flex-start !important;
        text-align: left !important;
        border: 1px solid transparent !important;
        color: var(--buddie-text) !important;
    }
    section[data-testid="stSidebar"] button[kind="tertiary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="baseButton-tertiary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-tertiary"]:hover {
        background: var(--buddie-tint) !important;
        border-color: var(--buddie-border) !important;
        color: var(--buddie-primary) !important;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        div[data-testid="stBottomBlockContainer"] {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .buddie-brand-title {
            font-size: 1.45rem;
        }
        .buddie-welcome-brand {
            font-size: 1.45rem;
        }
    }
    </style>
    """
)

init_chat_state()


def _get_client() -> ApiClient:
    """Session-scoped API client."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = ApiClient(
            config.api_base_url,
            timeout_seconds=config.request_timeout_seconds,
        )
    return st.session_state.api_client


def _answer_from_payload(mode: str, payload: dict[str, Any]) -> str:
    return format_assistant_answer(mode, payload)


def _agent_metadata() -> dict[str, Any] | None:
    """Session context for agent routing (verification + prior turn)."""
    meta: dict[str, Any] = {}
    eid = st.session_state.get("verified_employee_id")
    if eid:
        meta["employee_id"] = eid

    # Round-trip pending write drafts for human-in-the-loop confirmation.
    pending = st.session_state.get("pending_leave_request")
    if isinstance(pending, dict) and pending:
        meta["pending_leave_request"] = pending

    messages = st.session_state.get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            meta["last_assistant_message"] = content.strip()
            break

    return meta or None


def _render_header() -> None:
    """Top bar: accent + compact Buddie wordmark + actions.

    Uses native Streamlit heading/caption for the brand so layout height is
    measured correctly (custom ``st.html`` wordmarks were getting clipped).
    """
    st.html('<div class="buddie-header-accent" aria-hidden="true"></div>')
    left, right = st.columns([1.4, 1], vertical_alignment="top", gap="small")
    with left:
        st.markdown(
            '<p class="buddie-brand-title">'
            '<span class="buddie-brand-mark" aria-hidden="true"></span>Buddie</p>',
            unsafe_allow_html=True,
        )
        st.caption("AI Employee Assistant")
    with right:
        with st.container(horizontal=True, gap="small", horizontal_alignment="right"):
            render_verified_indicator()
            if st.session_state.get("verified_employee_id") and st.button(
                "Workspace",
                type="tertiary",
                icon=":material/view_sidebar:",
                key="open_employee_workspace",
                help="Open the employee workspace sidebar",
            ):
                request_workspace_sidebar_open()
                st.rerun()
            if st.button(
                "New conversation",
                type="tertiary",
                icon=":material/edit_square:",
                key="new_conversation",
                help="Start a fresh chat (keeps verification)",
            ):
                clear_conversation(clear_verification=False)
                st.rerun()
            if st.session_state.get("verified_employee_id") and st.button(
                "Sign out",
                type="tertiary",
                icon=":material/logout:",
                key="clear_verification",
                help="Clear employee verification",
            ):
                clear_conversation(clear_verification=True)
                st.rerun()
            # Developer entry lives in the employee sidebar once verified.
            if not st.session_state.get("verified_employee_id"):
                render_developer_menu(key="buddie_menu_header")


def _render_welcome() -> None:
    with st.container(horizontal_alignment="center", key="buddie_welcome"):
        st.html(
            """
            <div class="buddie-welcome" style="text-align:center;">
              <h3 style="color:#16323A; margin:0.75rem 0 0.3rem 0; font-weight:650;">
                How can I help you today?
              </h3>
              <p style="color:#64748B; margin:0; font-size:1rem;">
                Ask about leave, holidays, benefits, company policies,
                or your day-to-day work.
              </p>
            </div>
            """
        )


def _begin_verification(question: str) -> None:
    """Park the user question and ask for employee ID in natural language."""
    request_chat_scroll(force=True)
    append_message("user", question)
    append_message("assistant", VERIFY_PROMPT)
    st.session_state.pending_question = question
    st.session_state.awaiting_verification = True
    render_chat_autoscroll()


def _run_query(
    client: ApiClient,
    mode: str,
    question: str,
    *,
    already_logged: bool = False,
) -> None:
    """Call FastAPI and append the assistant reply (friendly errors only)."""
    # New user turn → resume following the latest message.
    request_chat_scroll(force=True)

    if not already_logged:
        append_message("user", question)
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.spinner("Buddie is thinking…"):
            try:
                if mode == "RAG":
                    payload = client.query_rag(question)
                else:
                    payload = client.query_agent(
                        question,
                        metadata=_agent_metadata(),
                    )
                store_response(mode, payload)
                answer = _answer_from_payload(mode, payload) or (
                    "I couldn't find an answer for that. "
                    "Try rephrasing, or ask about leave, benefits, or policies."
                )
                # Persist pending leave draft for confirm/cancel follow-ups.
                if mode == "AGENT" and isinstance(payload, dict):
                    agent_meta = payload.get("metadata") or {}
                    pending = agent_meta.get("pending_leave_request")
                    if isinstance(pending, dict) and pending.get(
                        "awaiting_confirmation"
                    ):
                        st.session_state.pending_leave_request = pending
                    else:
                        st.session_state.pending_leave_request = None
                st.markdown(answer)
                append_message("assistant", answer)
                st.session_state.last_error_technical = None
            except ApiClientError as exc:
                logger.warning("%s failure: %s", mode, exc.message)
                st.session_state.last_error_technical = technical_detail(exc)
                message = friendly_error(exc)
                st.error(message)
                append_message("assistant", message)

    render_chat_autoscroll()


def _handle_prompt(
    client: ApiClient,
    mode: str,
    prompt: str,
    *,
    already_logged: bool = False,
) -> None:
    question = prompt.strip()
    if not question:
        st.warning("Please enter a question.")
        return

    requires_id = needs_employee_verification(question)
    verified = bool(st.session_state.get("verified_employee_id"))
    if requires_id and not verified:
        _begin_verification(question)
        st.rerun()
        return

    _run_query(client, mode, question, already_logged=already_logged)


def _handle_demo(client: ApiClient, live: bool) -> None:
    with st.spinner("Running interview demo…"):
        try:
            result = client.run_demo(live=live)
            st.session_state.demo_result = result
            store_response("DEMO", result)
            answer = (result.get("agent_result") or {}).get("final_answer", "")
            decision = (result.get("quality_decision") or {}).get("status", "—")
            append_message(
                "assistant",
                f"Demo complete. Quality gate: **{decision}**\n\n{answer}",
            )
        except ApiClientError as exc:
            logger.warning("Demo failed: %s", exc.message)
            st.session_state.last_error_technical = technical_detail(exc)
            append_message("assistant", friendly_error(exc))


def main() -> None:
    """Streamlit entrypoint — Buddie employee assistant shell."""
    client = _get_client()

    if "query_mode" not in st.session_state:
        st.session_state.query_mode = "AGENT"
    if "developer_mode" not in st.session_state:
        st.session_state.developer_mode = False

    # Contextual employee workspace — only after verification.
    render_employee_workspace()

    _render_header()

    messages = st.session_state.messages
    empty_home = not messages and not st.session_state.awaiting_verification

    if empty_home:
        _render_welcome()
        render_quick_actions()

    render_chat_history()

    if st.session_state.awaiting_verification:
        render_verification_form(client)

    # After verify / sidebar layout shifts, restore latest-message visibility
    # when a prior turn requested follow (force flag survives st.rerun).
    if st.session_state.get("_buddie_scroll_force"):
        render_chat_autoscroll()

    query_mode = st.session_state.query_mode
    if query_mode not in {"RAG", "AGENT"}:
        query_mode = "AGENT"
    # Employees always get Buddie; RAG-only path is a developer demo control.
    if not st.session_state.get("developer_mode"):
        query_mode = "AGENT"

    queued_demo = st.session_state.pop("queued_demo", None)
    if queued_demo:
        _handle_demo(client, live=bool(queued_demo.get("live")))
        st.rerun()

    resume_after_verify = bool(st.session_state.pop("resume_after_verify", False))
    queued = st.session_state.pop("queued_prompt", None)
    chat_value = st.chat_input(
        "Ask Buddie anything about work, leave, benefits or policies...",
        submit_mode="disable",
    )
    prompt = queued or chat_value
    if prompt is not None:
        if st.session_state.awaiting_verification and queued is None:
            st.info(
                "Please verify your employee ID to continue.",
                icon=":material/badge:",
            )
        else:
            _handle_prompt(
                client,
                query_mode,
                str(prompt),
                already_logged=resume_after_verify,
            )

    # Technical console is opt-in only — never on the employee landing screen.
    if st.session_state.get("developer_mode"):
        st.space("medium")
        last = st.session_state.last_payload
        payload_mode = st.session_state.last_mode or "AGENT"
        selected = render_developer_console(
            client,
            last,
            payload_mode,
            query_mode=query_mode,
        )
        if selected in {"RAG", "AGENT"}:
            st.session_state.query_mode = selected


if __name__ == "__main__":
    main()
