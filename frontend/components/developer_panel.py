"""Interview / developer observability — hidden from the main Buddie UX."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import ApiClient
from frontend.components.debug_panel import render_debug_panel
from frontend.components.evaluation_panel import render_evaluation_panel
from frontend.components.execution_metadata import extract_execution_metadata
from frontend.components.reports_panel import render_reports_panel
from frontend.components.trace_panel import render_agent_flow, render_langsmith_panel


def render_developer_menu(*, key: str = "buddie_menu") -> None:
    """Discreet entry into Developer / Evaluation mode."""
    with st.popover(
        "⋮",
        help="Developer / Evaluation",
        key=key,
    ):
        st.caption("Advanced")
        if st.session_state.get("developer_mode"):
            if st.button(
                "Exit developer mode",
                icon=":material/close:",
                key=f"exit_developer_mode_{key}",
                width="stretch",
            ):
                st.session_state.developer_mode = False
                st.rerun()
        else:
            if st.button(
                "Developer / Evaluation",
                icon=":material/terminal:",
                key=f"enter_developer_mode_{key}",
                width="stretch",
            ):
                st.session_state.developer_mode = True
                st.rerun()


def render_developer_console(
    client: ApiClient,
    payload: dict[str, Any] | None,
    payload_mode: str,
    *,
    query_mode: str = "AGENT",
) -> str:
    """Engineering console for interview demos — only when developer mode is on.

    Args:
        client: FastAPI client for health / reports / demo.
        payload: Latest backend response (chat or demo).
        payload_mode: Mode of the stored payload (``RAG`` / ``AGENT`` / ``DEMO``).
        query_mode: Preferred query path for the next user message.

    Returns:
        Selected query mode (``RAG`` or ``AGENT``).
    """
    with st.container(border=True):
        st.html(
            """
            <div class="buddie-dev-shell">
              <p class="buddie-dev-title">
                <span class="buddie-dev-dot" aria-hidden="true"></span>Developer Mode
              </p>
              <p class="buddie-dev-caption">
                Execution metadata from the real agent turn.
                Hidden from the employee-facing Buddie screen.
              </p>
            </div>
            """
        )
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.caption("Ocean Lagoon evaluation console")
            if st.button(
                "Exit",
                type="tertiary",
                icon=":material/close:",
                key="exit_developer_banner",
            ):
                st.session_state.developer_mode = False
                st.rerun()

        default_label = "Knowledge only" if query_mode == "RAG" else "Buddie"
        mode_label = st.segmented_control(
            "Assistant path (demo)",
            options=["Buddie", "Knowledge only"],
            default=default_label,
            help=(
                "Buddie uses the agent path (tools + policies). "
                "Knowledge only uses the RAG query path. "
                "Employees always see Buddie — they never choose a path."
            ),
            key="buddie_mode_control",
        )
        selected_mode = "RAG" if mode_label == "Knowledge only" else "AGENT"

        health_col, demo_col = st.columns(2)
        with health_col:
            try:
                health = client.get_health()
                st.caption(
                    f"Backend `{health.get('status', 'ok')}` · "
                    f"v{health.get('version', '?')} · "
                    f"`{health.get('service', 'api')}`"
                )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Backend unreachable: {exc}")

        with demo_col:
            live_demo = st.checkbox("Live demo (needs API keys)", value=False)
            if st.button("Run canonical demo", key="run_demo_btn"):
                st.session_state.queued_demo = {"live": live_demo}

        last_error = st.session_state.get("last_error_technical")
        if last_error:
            st.markdown("**Last technical error**")
            st.code(str(last_error), language="text")

        verify = st.session_state.get("last_verify_result")
        if verify:
            st.markdown("**Employee verification result**")
            st.json(
                {
                    "verified": bool(verify.get("verified")),
                    "employee_id": verify.get("employee_id"),
                    "source": verify.get("source"),
                }
            )

        st.markdown("#### Execution metadata")
        _render_execution_metadata(payload, payload_mode)

        st.markdown("#### Execution trace")
        _render_execution_trace(payload, payload_mode)
        render_agent_flow(payload, payload_mode)

        st.markdown("#### Debug")
        render_debug_panel(payload, payload_mode)

        st.markdown("#### Evaluation")
        render_evaluation_panel(payload, payload_mode)

        st.markdown("#### Observability")
        render_langsmith_panel(payload, payload_mode)

        st.markdown("#### Reports")
        render_reports_panel(client)

        return selected_mode


def _kv(label: str, value: Any) -> None:
    display = "—" if value is None or value == "" else value
    st.markdown(f"**{label}**  \n`{display}`")


def _render_execution_metadata(
    payload: dict[str, Any] | None,
    mode: str,
) -> None:
    """Surface sanitized agent execution fields for interview demos."""
    if not payload:
        st.caption("Ask Buddie a question to populate execution metadata.")
        return

    snapshot = extract_execution_metadata(
        payload,
        mode,
        verify_result=st.session_state.get("last_verify_result"),
    )

    # ---- Request ----
    st.markdown("##### Request")
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        _kv("Original input", snapshot.get("original_input"))
    with col_b:
        _kv("Normalized input", snapshot.get("normalized_input"))
        routing = snapshot.get("routing_normalized_input")
        if routing and routing != snapshot.get("normalized_input"):
            st.caption(f"Routing normalize: `{routing}`")

    # ---- Agent decision ----
    st.markdown("##### Agent Decision")
    st.divider()
    intent = snapshot.get("detected_intent") or "—"
    route = snapshot.get("selected_route") or "—"
    st.markdown(f"**Intent**  \n`{intent}`")
    st.markdown(f"**Route**  \n`{route}`")

    required = bool(snapshot.get("verification_required"))
    status = str(snapshot.get("verification_status") or "").lower()
    verified_id = snapshot.get("verified_employee_id")
    if status == "verified" and verified_id:
        st.badge(
            f"Verified — {verified_id}",
            icon=":material/check_circle:",
            color="green",
        )
    elif not required or status == "not_required":
        st.badge("Not required", icon=":material/remove:", color="gray")
    elif required:
        st.badge("Verification required", icon=":material/badge:", color="orange")
    else:
        st.caption(f"Verification: `{status or '—'}`")

    # ---- Retrieval ----
    st.markdown("##### Retrieval")
    st.divider()
    if snapshot.get("rag_used"):
        st.badge("RAG Used", icon=":material/check:", color="green")
        sources = snapshot.get("retrieved_sources") or []
        if sources:
            st.markdown("**Sources**")
            st.code("\n".join(f"- {src}" for src in sources), language="text")
        top_k = snapshot.get("top_k")
        if top_k is not None:
            _kv("Top-K", top_k)
        scores = snapshot.get("retrieval_scores") or []
        if scores:
            formatted = ", ".join(f"{score:.3f}" for score in scores[:8])
            _kv("Scores", formatted)
        query = snapshot.get("retrieval_query")
        if query:
            _kv("Retrieval query", query)
    else:
        st.markdown("**RAG used:** No")

    # ---- Tools ----
    st.markdown("##### Tools")
    st.divider()
    mcp = snapshot.get("mcp") if isinstance(snapshot.get("mcp"), dict) else {}
    if mcp.get("used"):
        st.markdown("**Protocol**  \n`MCP`")
        if mcp.get("connected"):
            st.badge("MCP Connected", icon=":material/check_circle:", color="green")
        else:
            st.badge("MCP Unavailable", icon=":material/error:", color="orange")
        transport = mcp.get("transport")
        if transport:
            st.caption(f"Transport: `{transport}`")
        mcp_latency = mcp.get("mcp_latency_ms")
        if mcp_latency is not None:
            st.caption(f"MCP latency: `{mcp_latency}` ms")

    tools = snapshot.get("tools_invoked") or []
    if not tools:
        st.markdown("**Tools:** None")
    else:
        for index, tool in enumerate(tools, start=1):
            if not isinstance(tool, dict):
                continue
            name = tool.get("tool_name") or "tool"
            status_raw = str(tool.get("status") or "—")
            ok = status_raw.lower() in {
                "ok",
                "success",
                "succeeded",
                "completed",
            }
            with st.container(border=True):
                st.markdown(f"**{index}. `{name}`**")
                if ok:
                    st.badge(
                        "Tool Success",
                        icon=":material/check_circle:",
                        color="green",
                    )
                elif tool.get("error"):
                    st.badge("Tool Failed", icon=":material/error:", color="red")
                else:
                    st.caption(f"Status: `{status_raw}`")
                protocol = tool.get("protocol")
                if protocol:
                    st.caption(f"Protocol: `{protocol}`")
                args = tool.get("arguments") or {}
                if args:
                    st.caption("Arguments")
                    st.json(args)
                summary = tool.get("result_summary")
                if summary:
                    st.caption(f"Result: {summary}")
                tool_latency = tool.get("latency_ms")
                if tool_latency is not None:
                    st.caption(f"Latency: `{tool_latency}` ms")
                mcp_tool_latency = tool.get("mcp_latency_ms")
                if mcp_tool_latency is not None:
                    st.caption(f"MCP latency: `{mcp_tool_latency}` ms")
                if tool.get("error"):
                    st.caption("Error: tool failed (details withheld)")

    # ---- Execution ----
    st.markdown("##### Execution")
    st.divider()
    exec_a, exec_b = st.columns(2)
    with exec_a:
        latency = snapshot.get("latency_ms")
        _kv(
            "Latency (ms)",
            f"{latency:.1f}" if isinstance(latency, (int, float)) else latency,
        )
        model = snapshot.get("model")
        if model:
            _kv("Model", model)
    with exec_b:
        run_url = snapshot.get("langsmith_run_url")
        if run_url:
            st.link_button(
                "LangSmith trace",
                str(run_url),
                icon=":material/open_in_new:",
                type="secondary",
            )
        else:
            st.caption("LangSmith URL not available on this response.")


def _render_execution_trace(
    payload: dict[str, Any] | None,
    mode: str,
) -> None:
    """Compact numbered steps for interview walkthroughs."""
    if not payload:
        st.caption("Ask Buddie a question to populate the execution trace.")
        return

    steps: list[tuple[str, str]] = []

    verify = st.session_state.get("last_verify_result")
    if verify and verify.get("verified"):
        steps.append(("verify_employee", "VERIFIED"))

    agent: dict[str, Any] | None = None
    if mode == "DEMO":
        agent = payload.get("agent_result") if isinstance(payload, dict) else None
    elif mode == "AGENT" and isinstance(payload, dict):
        agent = payload

    if isinstance(agent, dict):
        tools = agent.get("tool_executions") or []
        for tool in tools:
            name = str(tool.get("tool_name") or "tool")
            status = tool.get("status", "SUCCESS")
            if isinstance(status, dict):
                status = status.get("value", status)
            status_text = str(status).upper()
            if status_text.lower() in {"ok", "success", "succeeded", "completed"}:
                status_text = "SUCCESS"
            steps.append((name, status_text))

        ctx = agent.get("evaluation_context") or {}
        chunks = ctx.get("retrieved_chunks") or []
        docs = ctx.get("retrieved_documents") or []
        if chunks or docs:
            steps.append(("RAG retrieval", "SUCCESS"))
        if agent.get("final_answer"):
            steps.append(("Answer generation", "SUCCESS"))
    elif mode == "RAG" and isinstance(payload, dict):
        docs = payload.get("retrieved_documents") or []
        steps.append(
            ("RAG retrieval", "SUCCESS" if docs else "NO DOCUMENTS"),
        )
        if payload.get("answer"):
            steps.append(("Answer generation", "SUCCESS"))

    if not steps:
        st.caption("No tool or retrieval steps on the latest response yet.")
        return

    lines = []
    for index, (name, status) in enumerate(steps, start=1):
        mark = "✓" if status in {"SUCCESS", "VERIFIED"} else "·"
        lines.append(f"{index}. {name}\n   {mark} {status}")
    st.code("\n\n".join(lines), language="text")
