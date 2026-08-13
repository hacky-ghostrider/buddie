"""Agent execution flow + LangSmith trace panel."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_agent_flow(payload: dict[str, Any] | None, mode: str) -> None:
    """Visualize planner → tools → answer using backend payloads only."""
    with st.expander("Agent Flow", expanded=mode in {"AGENT", "DEMO"}):
        if not payload or mode == "RAG":
            st.info("Switch to Agent mode or Run Demo to see the execution flow.")
            return

        agent = payload.get("agent_result") if mode == "DEMO" else payload
        if not isinstance(agent, dict):
            st.warning("Unexpected agent payload.")
            return

        planner = agent.get("planner_output") or agent.get("planner_decision") or {}
        tools = agent.get("tool_executions") or []
        tv = None
        if mode == "DEMO":
            report = payload.get("evaluation_report") or {}
            tv = report.get("tool_validation") or agent.get("tool_validation")
        else:
            tv = agent.get("tool_validation")

        expected = (tv or {}).get("expected_tools") or planner.get("required_tools") or []
        actual = (tv or {}).get("actual_tools") or [t.get("tool_name") for t in tools]

        flow_lines = [
            "User Question",
            "↓",
            "LangGraph",
            "↓",
            "Planner",
            "↓",
            "Tool Selection",
        ]
        meta = agent.get("metadata") or {}
        mcp = meta.get("mcp") if isinstance(meta.get("mcp"), dict) else {}
        if mcp.get("used"):
            flow_lines.extend(["↓", "MCP"])
            if mcp.get("connected"):
                flow_lines.extend(["↓", "MCP Connected ✓"])
        for name in actual:
            protocol = ""
            for tool in tools:
                if isinstance(tool, dict) and tool.get("tool_name") == name:
                    protocol = tool.get("protocol") or (tool.get("trace_metadata") or {}).get(
                        "protocol"
                    )
                    break
            # Prefer metadata.tools_invoked protocol when present.
            for item in meta.get("tools_invoked") or []:
                if isinstance(item, dict) and item.get("tool_name") == name:
                    protocol = item.get("protocol") or protocol
                    break
            label = f"MCP {name}" if str(protocol).upper() == "MCP" else str(name)
            flow_lines.extend(["↓", label])
        flow_lines.extend(["↓", "Final Answer"])
        st.code("\n".join(flow_lines), language="text")

        rationale = planner.get("rationale") or planner.get("reasoning") or ""
        if rationale:
            st.caption(f"Planner: {rationale}")

        st.markdown(f"**Expected tools:** `{expected}`")
        st.markdown(f"**Actual tools:** `{actual}`")
        st.markdown(f"**Tool execution order:** `{[t.get('tool_name') for t in tools]}`")

        if tv is not None:
            passed = tv.get("passed")
            st.markdown(f"**Tool validation:** {'PASS' if passed else 'FAIL'}")
        else:
            st.caption("Tool validation report not present on this response.")


def render_langsmith_panel(payload: dict[str, Any] | None, mode: str) -> None:
    """Show LangSmith trace metadata provided by the backend (no direct LS calls)."""
    with st.expander("LangSmith Trace", expanded=False):
        if not payload:
            st.info("Run an Agent query or Demo to see trace metadata.")
            return

        if mode == "DEMO":
            agent = payload.get("agent_result") or {}
            report = payload.get("evaluation_report") or {}
            trace_id = agent.get("trace_id") or report.get("langsmith_trace_id")
            run_id = agent.get("run_id") or report.get("langsmith_run_id")
            run_url = agent.get("run_url") or report.get("langsmith_run_url")
        elif mode == "AGENT":
            ctx = payload.get("evaluation_context") or {}
            trace_id = payload.get("trace_id") or ctx.get("langsmith_trace_id")
            run_id = payload.get("run_id") or ctx.get("langsmith_run_id")
            run_url = payload.get("run_url") or ctx.get("langsmith_run_url")
        else:
            st.caption("RAG responses may not include LangSmith agent traces.")
            return

        st.markdown(f"**Trace ID:** `{trace_id or '—'}`")
        st.markdown(f"**Run ID:** `{run_id or '—'}`")
        if run_url:
            st.link_button(
                "View LangSmith Trace",
                run_url,
                icon=":material/open_in_new:",
                type="secondary",
            )
        else:
            st.caption(
                "No live LangSmith URL (NoOpTracer / tracing disabled). "
                "Enable ENABLE_LANGSMITH on the backend for a clickable link."
            )
