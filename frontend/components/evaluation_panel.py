"""Expandable evaluation panel — displays backend EvaluationReport / QualityDecision."""

from __future__ import annotations

from typing import Any

import streamlit as st

_METRIC_ALIASES = {
    "faithfulness": "Faithfulness",
    "hallucination": "Hallucination",
    "answer_relevancy": "Answer Relevancy",
    "relevancy": "Answer Relevancy",
    "contextual_precision": "Context Precision",
    "context_precision": "Context Precision",
    "contextual_recall": "Context Recall",
    "context_recall": "Context Recall",
}


def render_evaluation_panel(
    payload: dict[str, Any] | None,
    mode: str,
) -> None:
    """Show DeepEval metrics, tool validation, and quality gate from backend data."""
    with st.expander("Evaluation", expanded=False):
        if not payload:
            st.info("Run a query or demo to see evaluation results.")
            return

        if mode == "DEMO":
            report = payload.get("evaluation_report") or {}
            decision = payload.get("quality_decision") or {}
            agent = payload.get("agent_result") or {}
            _render_metrics(report)
            _render_tool_validation(
                report.get("tool_validation") or agent.get("tool_validation")
            )
            _render_quality_gate(decision, report)
            return

        if mode == "AGENT":
            tv = payload.get("tool_validation")
            _render_tool_validation(tv)
            st.caption(
                "DeepEval metrics and quality gates are available via **Run Demo** "
                "(canonical scenario) or evaluation reports."
            )
            return

        st.caption(
            "RAG mode returns retrieval + generation observability. "
            "Full DeepEval / quality-gate panels appear after **Run Demo**."
        )


def _render_metrics(report: dict[str, Any]) -> None:
    metrics = report.get("metrics") or []
    if not metrics:
        st.write("No metrics in this response.")
        return

    st.markdown("**DeepEval / Metrics**")
    rows = []
    for m in metrics:
        name = m.get("name", "")
        label = _METRIC_ALIASES.get(name, name)
        rows.append(
            {
                "Metric": label,
                "Score": round(float(m.get("score", 0.0)), 3),
                "Passed": m.get("passed"),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.markdown(f"**Overall score:** {report.get('overall_score', '—')}")


def _render_tool_validation(tv: dict[str, Any] | None) -> None:
    st.markdown("**Tool Validation**")
    if not tv:
        st.write("Not available for this response.")
        return
    passed = tv.get("passed")
    badge = "PASS" if passed else "FAIL"
    st.markdown(f"Result: **{badge}**")
    st.markdown(f"- Expected tools: `{tv.get('expected_tools', [])}`")
    st.markdown(f"- Actual tools: `{tv.get('actual_tools', [])}`")
    failures = tv.get("failures") or []
    if failures:
        st.markdown("Failures:")
        for f in failures:
            st.write(f"- {f}")


def _render_quality_gate(decision: dict[str, Any], report: dict[str, Any]) -> None:
    st.markdown("**Quality Gate**")
    if not decision:
        st.write("Not available.")
        return
    status = decision.get("status", "—")
    if isinstance(status, dict):
        status = status.get("value", status)
    color = {"PASS": "green", "WARNING": "orange", "FAIL": "red"}.get(str(status), "gray")
    st.markdown(f":{color}[**{status}**] — {decision.get('reason', '')}")
    st.markdown(f"- Overall score: `{decision.get('overall_score', report.get('overall_score'))}`")
    st.markdown(f"- Failed rules: `{decision.get('failed_rules') or []}`")
    st.markdown(f"- Warnings: `{decision.get('warnings') or []}`")
    recs = decision.get("recommendations") or []
    if recs:
        st.markdown("Recommendations:")
        for rec in recs[:5]:
            st.write(f"- {rec.get('message', rec)}")
