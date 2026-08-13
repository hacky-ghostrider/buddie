"""Report browser — lists / views existing backend report artifacts."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import ApiClient, ApiClientError


def render_reports_panel(client: ApiClient) -> None:
    """Allow viewing the latest evaluation / quality / benchmark reports."""
    with st.expander("Reports", expanded=False):
        st.caption("Reuses existing report files — no second reporting system.")
        kind = st.selectbox(
            "Report kind",
            options=[
                "demo_evaluation",
                "demo_quality",
                "demo_benchmark",
                "evaluation",
                "quality",
                "benchmark",
            ],
            index=0,
        )
        if st.button("Refresh report list", key="refresh_reports"):
            st.session_state.pop("report_list", None)

        try:
            listing = client.list_reports(kind=kind)
            files = listing.get("files") or []
        except ApiClientError as exc:
            st.error(exc.message)
            return

        if not files:
            st.info(f"No {kind} reports found yet. Run Demo to generate artifacts.")
            return

        names = [f.get("name") for f in files if f.get("name")]
        selected = st.selectbox("File", options=names, key=f"report_file_{kind}")
        if not selected:
            return
        if st.button("View report", key=f"view_report_{kind}"):
            try:
                content = client.get_report(kind, selected)
            except ApiClientError as exc:
                st.error(exc.message)
                return
            _render_content(content)


def _render_content(payload: dict[str, Any]) -> None:
    content = payload.get("content")
    content_type = payload.get("content_type", "")
    st.markdown(f"**Path:** `{payload.get('path', '')}`")
    if content_type == "application/json" and isinstance(content, dict):
        st.json(content)
    elif content_type == "text/html" and isinstance(content, str):
        st.html(content)
    else:
        st.code(str(content), language="text")
