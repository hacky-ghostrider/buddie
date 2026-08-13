"""Chat history and session helpers for the Buddie Streamlit client."""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

# Near-bottom threshold (px) before we treat the user as reading history.
_SCROLL_NEAR_BOTTOM_PX = 140


def init_chat_state() -> None:
    """Ensure chat / verification session keys exist."""
    defaults: dict[str, Any] = {
        "messages": [],
        "last_payload": None,
        "last_mode": "AGENT",
        "demo_result": None,
        "verified_employee_id": None,
        "awaiting_verification": False,
        "pending_question": None,
        "queued_prompt": None,
        "queued_demo": None,
        "query_mode": "AGENT",
        "developer_mode": False,
        "resume_after_verify": False,
        "last_error_technical": None,
        "last_verify_result": None,
        "verify_error": None,
        "pending_leave_request": None,
        "_buddie_scroll_nonce": 0,
        "_buddie_scroll_force": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation(*, clear_verification: bool = False) -> None:
    """Reset chat history and last response payloads."""
    st.session_state.messages = []
    st.session_state.last_payload = None
    st.session_state.demo_result = None
    st.session_state.awaiting_verification = False
    st.session_state.pending_question = None
    st.session_state.queued_prompt = None
    st.session_state.last_error_technical = None
    st.session_state.verify_error = None
    st.session_state.pending_leave_request = None
    st.session_state._buddie_scroll_force = False
    if clear_verification:
        st.session_state.verified_employee_id = None
        st.session_state.last_verify_result = None
        st.session_state._buddie_open_sidebar = False
        st.session_state._buddie_sidebar_opened_once = False
        st.session_state._buddie_sidebar_open_nonce = 0


def render_chat_history() -> None:
    """Render prior user / assistant messages."""
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        avatar = (
            ":material/person:"
            if role == "user"
            else ":material/smart_toy:"
        )
        with st.chat_message(role, avatar=avatar):
            st.markdown(message.get("content", ""))


def append_message(role: str, content: str) -> None:
    """Append one chat turn to session history."""
    st.session_state.messages.append({"role": role, "content": content})


def store_response(mode: str, payload: dict[str, Any]) -> None:
    """Remember the latest backend payload for debug / eval panels."""
    st.session_state.last_mode = mode
    st.session_state.last_payload = payload


def request_chat_scroll(*, force: bool = True) -> None:
    """Mark that the chat viewport should follow the latest message.

    ``force=True`` is used when the user just sent a message (or verification
    resumed a pending ask) so we resume bottom-following even if they had
    scrolled upward earlier. Soft follow uses a near-bottom check in JS.
    """
    st.session_state._buddie_scroll_force = bool(
        force or st.session_state.get("_buddie_scroll_force")
    )
    st.session_state._buddie_scroll_nonce = (
        int(st.session_state.get("_buddie_scroll_nonce") or 0) + 1
    )


def render_chat_autoscroll() -> None:
    """Scroll Streamlit's app scroll container to the latest chat turn.

    Targets ``[data-testid="stAppScrollToBottomContainer"]`` — the real
    overflow scroller Streamlit uses with ``st.chat_input`` — not
    ``window``. Respects intentional upward history reading unless a forced
    follow was requested (new user submit).
    """
    nonce = int(st.session_state.get("_buddie_scroll_nonce") or 0)
    if nonce <= 0:
        return
    force = bool(st.session_state.pop("_buddie_scroll_force", False))
    # Keep nonce so a soft re-render can still re-run when force is set again.
    threshold = _SCROLL_NEAR_BOTTOM_PX
    components.html(
        f"""
        <script>
        (function () {{
          const nonce = {nonce};
          const force = {str(force).lower()};
          const threshold = {threshold};
          const doc = window.parent.document;
          const scroller = doc.querySelector(
            '[data-testid="stAppScrollToBottomContainer"]'
          );
          if (!scroller) {{
            return;
          }}

          const distanceFromBottom = () =>
            scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;

          const shouldFollow = () =>
            force || distanceFromBottom() <= threshold;

          const scrollLatest = () => {{
            if (!shouldFollow()) {{
              return;
            }}
            scroller.scrollTo({{
              top: scroller.scrollHeight,
              behavior: force ? "smooth" : "auto",
            }});
          }};

          // Run after Streamlit paints the newest chat message(s).
          scrollLatest();
          requestAnimationFrame(scrollLatest);
          setTimeout(scrollLatest, 60);
          setTimeout(scrollLatest, 180);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


__all__ = [
    "init_chat_state",
    "clear_conversation",
    "render_chat_history",
    "append_message",
    "store_response",
    "request_chat_scroll",
    "render_chat_autoscroll",
]
