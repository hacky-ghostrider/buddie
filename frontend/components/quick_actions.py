"""Clickable quick actions that populate the conversation — no navigation."""

from __future__ import annotations

import streamlit as st

# Primary employee tasks — keep the empty state calm (≈4 starters).
PRIMARY_ACTIONS: dict[str, str] = {
    "My leave": "How many vacation days do I have left?",
    "Leave history": "How many vacation days did I take last year?",
    "Upcoming holidays": "What are the upcoming company holidays?",
    "Pending actions": "What are my pending tasks?",
}

# Available via conversation or a discreet secondary menu — not the main grid.
SECONDARY_ACTIONS: dict[str, str] = {
    "My benefits": "What benefits am I enrolled in?",
    "Company policies": "Summarize the company vacation policy.",
}


def _queue_prompt(question: str, *, clear_key: str | None = None) -> None:
    st.session_state.queued_prompt = question
    if clear_key:
        st.session_state.pop(clear_key, None)
    st.rerun()


def render_quick_actions() -> None:
    """Show a small set of conversation starters under the welcome copy."""
    with st.container(horizontal_alignment="center"):
        st.html(
            """
            <p style="margin:0 0 0.35rem 0; color:#0F766E; font-size:0.85rem;
               font-weight:600; letter-spacing:0.02em; text-align:center;">
              Try asking about
            </p>
            """
        )
        selected = st.pills(
            "Quick actions",
            options=list(PRIMARY_ACTIONS.keys()),
            selection_mode="single",
            label_visibility="collapsed",
            key="buddie_quick_actions",
        )
        if selected:
            _queue_prompt(PRIMARY_ACTIONS[selected], clear_key="buddie_quick_actions")

        with st.popover("More topics", type="tertiary", icon=":material/add:"):
            st.caption("Other common questions")
            for label, question in SECONDARY_ACTIONS.items():
                if st.button(label, key=f"more_topic_{label}", width="stretch"):
                    _queue_prompt(question)
