"""Expandable debug panel — displays backend observability fields only."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_debug_panel(payload: dict[str, Any] | None, mode: str) -> None:
    """Show debug metadata returned by RAG / Agent / Demo responses."""
    with st.expander("Debug", expanded=False):
        if not payload:
            st.info("Run a query to see debug details.")
            return

        if mode == "DEMO":
            agent = payload.get("agent_result") or {}
            report = payload.get("evaluation_report") or {}
            _render_agent_debug(agent, report=report, extra=payload)
            return

        if mode == "AGENT":
            _render_agent_debug(payload)
            return

        _render_rag_debug(payload)


def _render_rag_debug(payload: dict[str, Any]) -> None:
    gen = payload.get("generation_metadata") or {}
    latency = payload.get("latency") or {}
    docs = payload.get("retrieved_documents") or []

    st.markdown(f"**Question:** {payload.get('question', '—')}")
    st.markdown(f"**Model:** {gen.get('model', '—')}")
    st.markdown(f"**Correlation ID:** `{payload.get('correlation_id', '—')}`")
    st.markdown(
        f"**Latency (ms):** retrieval={latency.get('retrieval_ms', '—')} · "
        f"prompt={latency.get('prompt_build_ms', '—')} · "
        f"llm={latency.get('llm_ms', '—')} · "
        f"total={latency.get('total_ms', '—')}"
    )
    tokens = {
        k: gen.get(k)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if gen.get(k) is not None
    }
    if tokens:
        st.markdown(f"**Token usage:** `{tokens}`")
    prompt_version = gen.get("prompt_version") or gen.get("template_version")
    if prompt_version:
        st.markdown(f"**Prompt version:** {prompt_version}")

    st.markdown(f"**Retrieved documents:** {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        meta = doc.get("metadata") or {}
        source = meta.get("source") or meta.get("file_name") or doc.get("id", "—")
        score = doc.get("score")
        with st.container():
            st.caption(f"[{i}] {source} · score={score}")
            st.text((doc.get("text") or "")[:500])


def _render_agent_debug(
    payload: dict[str, Any],
    *,
    report: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    ctx = payload.get("evaluation_context") or {}
    tools = payload.get("tool_executions") or []
    report = report or {}

    st.markdown(f"**Question:** {payload.get('question', '—')}")
    st.markdown(f"**Model:** {ctx.get('model') or '—'}")
    st.markdown(f"**Correlation ID:** `{payload.get('correlation_id', '—')}`")
    st.markdown(f"**Trace ID:** `{payload.get('trace_id') or '—'}`")
    st.markdown(f"**Latency (ms):** {payload.get('latency_ms', '—')}")

    token_usage = ctx.get("token_usage") or report.get("token_usage") or {}
    if token_usage:
        st.markdown(f"**Token usage:** `{token_usage}`")
    if ctx.get("prompt_version"):
        st.markdown(f"**Prompt version:** {ctx.get('prompt_version')}")

    chunks = ctx.get("retrieved_chunks") or []
    docs = ctx.get("retrieved_documents") or []
    st.markdown(f"**Retrieved documents:** {len(docs)}")
    st.markdown(f"**Retrieved chunks:** {len(chunks)}")
    for i, chunk in enumerate(chunks[:5], start=1):
        st.caption(f"Chunk {i}")
        st.text(str(chunk)[:400])

    st.markdown("**Tool executions**")
    if not tools:
        st.write("None")
    else:
        for index, tool in enumerate(tools, start=1):
            status = tool.get("status", "—")
            if isinstance(status, dict):
                status = status.get("value", status)
            ok = str(status).lower() in {"ok", "success", "succeeded", "completed"}
            mark = "✓" if ok else str(status)
            name = tool.get("tool_name") or "unknown"
            args = tool.get("arguments", {})
            # Avoid dumping oversized / sensitive payloads in the debug pane.
            if isinstance(args, dict):
                safe_args = {
                    k: ("[redacted]" if any(
                        t in str(k).lower()
                        for t in ("key", "token", "secret", "password")
                    ) else v)
                    for k, v in args.items()
                }
            else:
                safe_args = args
            st.markdown(
                f"{index}. `{name}` {mark} · "
                f"latency_ms={tool.get('latency_ms', '—')} · "
                f"args=`{safe_args}`"
            )

    if extra:
        st.markdown(f"**Demo mode:** {extra.get('mode', '—')}")
        st.markdown(f"**Scenario:** `{extra.get('scenario_id', '—')}`")
