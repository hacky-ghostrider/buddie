"""Authenticated employee workspace sidebar for Buddie.

Shown only after successful verification. Quick-access actions queue the
existing agent capabilities (leave balance, leave history, holidays, etc.)
through the normal chat + verification path — no parallel auth or fake data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st
import streamlit.components.v1 as components

from frontend.components.developer_panel import render_developer_menu


@dataclass(frozen=True)
class WorkspaceAction:
    """One quick-access entry that maps to an existing Buddie capability."""

    label: str
    question: str
    tool_name: str
    requires_verification: bool
    icon: str
    key: str


# Capability map — questions match existing planner/router intents.
QUICK_ACCESS_ACTIONS: tuple[WorkspaceAction, ...] = (
    WorkspaceAction(
        label="My Leave",
        question="How many vacation days do I have left?",
        tool_name="get_leave_balance",
        requires_verification=True,
        icon=":material/beach_access:",
        key="workspace_my_leave",
    ),
    WorkspaceAction(
        label="Leave History",
        question="Show my leave history",
        tool_name="get_leave_history",
        requires_verification=True,
        icon=":material/history:",
        key="workspace_leave_history",
    ),
    WorkspaceAction(
        label="Upcoming Holidays",
        question="What are the upcoming company holidays?",
        tool_name="get_upcoming_holidays",
        requires_verification=False,
        icon=":material/celebration:",
        key="workspace_upcoming_holidays",
    ),
    WorkspaceAction(
        label="Pending Actions",
        question="What are my pending tasks?",
        tool_name="get_pending_actions",
        requires_verification=True,
        icon=":material/pending_actions:",
        key="workspace_pending_actions",
    ),
)

OPTIONAL_ACTIONS: tuple[WorkspaceAction, ...] = (
    WorkspaceAction(
        label="Employee Profile",
        question="Show my profile",
        tool_name="get_employee_profile",
        requires_verification=True,
        icon=":material/badge:",
        key="workspace_employee_profile",
    ),
)


def is_employee_verified() -> bool:
    """True when session holds a verified employee id."""
    eid = st.session_state.get("verified_employee_id")
    return bool(eid and str(eid).strip())


def verified_employee_id() -> str | None:
    """Return the session verified employee id, or None."""
    eid = st.session_state.get("verified_employee_id")
    if not eid:
        return None
    cleaned = str(eid).strip()
    return cleaned or None


def employee_workspace_visible() -> bool:
    """Employee-specific sidebar content is available only after verification."""
    return is_employee_verified()


def sidebar_state_for_session() -> str:
    """Streamlit sidebar chrome: expanded after verify, collapsed before."""
    return "expanded" if is_employee_verified() else "collapsed"


def request_workspace_sidebar_open() -> None:
    """Ask the next run to open the employee workspace sidebar."""
    st.session_state._buddie_open_sidebar = True


def clear_workspace_sidebar_flags() -> None:
    """Reset open/once flags (used on sign-out)."""
    st.session_state._buddie_open_sidebar = False
    st.session_state._buddie_sidebar_opened_once = False
    st.session_state._buddie_sidebar_open_nonce = 0


def queue_workspace_action(action: WorkspaceAction) -> None:
    """Queue an existing capability via chat, enforcing verification when needed.

    Protected actions without a verified session start the verification flow
    instead of calling tools with a spoofed employee id.
    """
    if action.requires_verification and not is_employee_verified():
        st.session_state.pending_question = action.question
        st.session_state.awaiting_verification = True
        st.session_state.queued_prompt = None
        return

    st.session_state.queued_prompt = action.question


def action_for_label(label: str) -> WorkspaceAction | None:
    """Look up a workspace action by display label."""
    for action in (*QUICK_ACCESS_ACTIONS, *OPTIONAL_ACTIONS):
        if action.label == label:
            return action
    return None


def render_employee_workspace(
    *,
    on_developer_menu: Callable[[], None] | None = None,
) -> None:
    """Render the collapsible left employee workspace when verified.

    Before verification this is a no-op so Buddie stays a normal chat
    experience with no employee-specific sidebar content.
    """
    if not employee_workspace_visible():
        return

    eid = verified_employee_id()
    assert eid is not None  # guarded by employee_workspace_visible()

    with st.sidebar:
        st.markdown(
            '<p class="buddie-ws-brand">'
            '<span class="buddie-brand-mark" aria-hidden="true"></span>Buddie</p>',
            unsafe_allow_html=True,
        )
        st.caption("Employee workspace")

        st.markdown('<p class="buddie-ws-section">EMPLOYEE</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="buddie-ws-identity">'
            f'<div class="buddie-ws-eid">{eid}</div>'
            f'<div class="buddie-ws-verified">✓ Verified</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="buddie-ws-section">QUICK ACCESS</p>',
            unsafe_allow_html=True,
        )
        for action in QUICK_ACCESS_ACTIONS:
            if st.button(
                action.label,
                icon=action.icon,
                key=action.key,
                type="tertiary",
                width="stretch",
                help=f"Open {action.label.lower()} via Buddie",
            ):
                queue_workspace_action(action)
                st.rerun()

        st.markdown('<p class="buddie-ws-section">OPTIONAL</p>', unsafe_allow_html=True)
        for action in OPTIONAL_ACTIONS:
            if st.button(
                action.label,
                icon=action.icon,
                key=action.key,
                type="tertiary",
                width="stretch",
                help=f"Open {action.label.lower()} via Buddie",
            ):
                queue_workspace_action(action)
                st.rerun()

        st.divider()
        st.caption("Advanced")
        if on_developer_menu is not None:
            on_developer_menu()
        else:
            render_developer_menu(key="buddie_menu_workspace")

    # Open on verify / explicit reopen. Keep expand control usable after collapse.
    ensure_workspace_sidebar_open()


def ensure_workspace_sidebar_open() -> None:
    """Keep the employee sidebar recoverable across Cursor + Chrome.

    Behavior:
    - Force-open after verify / explicit Workspace click (flag).
    - Force-open once on first verified paint (Chrome often stays collapsed).
    - After the user collapses, do **not** yank it back open on every rerun.
    - Always restyle/reposition Streamlit's expand control so collapse is reversible.
    """
    force = bool(st.session_state.pop("_buddie_open_sidebar", False))
    opened_once = bool(st.session_state.get("_buddie_sidebar_opened_once"))
    should_click_open = force or not opened_once
    if should_click_open:
        st.session_state._buddie_sidebar_opened_once = True

    nonce = int(st.session_state.get("_buddie_sidebar_open_nonce") or 0) + 1
    st.session_state._buddie_sidebar_open_nonce = nonce
    components.html(
        f"""
        <script>
        (function () {{
          const nonce = {nonce};
          const shouldOpen = {str(should_click_open).lower()};
          const doc = window.parent.document;

          const styleExpandControl = () => {{
            const btn = doc.querySelector('[data-testid="stExpandSidebarButton"]');
            if (!btn) {{
              return null;
            }}
            const header = doc.querySelector('header[data-testid="stHeader"]');
            if (header) {{
              header.style.setProperty('overflow', 'visible', 'important');
              header.style.setProperty('height', '0px', 'important');
              header.style.setProperty('min-height', '0px', 'important');
            }}
            btn.style.setProperty('position', 'fixed', 'important');
            btn.style.setProperty('top', '0.7rem', 'important');
            btn.style.setProperty('left', '0.7rem', 'important');
            btn.style.setProperty('z-index', '1000000', 'important');
            btn.style.setProperty('display', 'inline-flex', 'important');
            btn.style.setProperty('visibility', 'visible', 'important');
            btn.style.setProperty('opacity', '1', 'important');
            btn.style.setProperty('pointer-events', 'auto', 'important');
            btn.style.setProperty('width', '2.25rem', 'important');
            btn.style.setProperty('height', '2.25rem', 'important');
            return btn;
          }};

          const openSidebar = () => {{
            const expandBtn = styleExpandControl();
            if (!shouldOpen) {{
              return;
            }}
            if (expandBtn) {{
              expandBtn.click();
              return;
            }}
            const sidebar = doc.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) {{
              return;
            }}
            const rect = sidebar.getBoundingClientRect();
            if (rect.width < 120) {{
              const fallback = doc.querySelector(
                'button[kind="headerNoPadding"], button[data-testid="baseButton-headerNoPadding"]'
              );
              if (fallback) {{
                fallback.click();
              }}
            }}
          }};

          openSidebar();
          requestAnimationFrame(openSidebar);
          setTimeout(openSidebar, 50);
          setTimeout(openSidebar, 200);
          setTimeout(openSidebar, 500);
          // Keep expand control recoverable even after a later manual collapse.
          setTimeout(styleExpandControl, 800);
          setTimeout(styleExpandControl, 1600);
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


__all__ = [
    "OPTIONAL_ACTIONS",
    "QUICK_ACCESS_ACTIONS",
    "WorkspaceAction",
    "action_for_label",
    "clear_workspace_sidebar_flags",
    "employee_workspace_visible",
    "ensure_workspace_sidebar_open",
    "is_employee_verified",
    "queue_workspace_action",
    "render_employee_workspace",
    "request_workspace_sidebar_open",
    "sidebar_state_for_session",
    "verified_employee_id",
]
