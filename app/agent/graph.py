"""LangGraph agent graph — planner → router → finalize.

Modular graph construction keeps nodes independently testable and allows
future Sprint 12 features (memory, supervisor, MCP) to add nodes without
rewriting evaluation seams.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.planner import Planner
from app.agent.router import ToolRouter, executions_from_state
from app.agent.state import AgentState
from app.agent.tools.base import ToolRegistry
from app.evaluation.context import EvaluationContext
from app.retrieval.models import RetrievedDocument

logger = logging.getLogger(__name__)


def _finalize_node(state: AgentState) -> dict[str, Any]:
    """Build ``EvaluationContext`` from completed tool executions.

    This node never bypasses ``ToolExecution`` — it only aggregates state
    into the Sprint 10.2 evaluation DTO.
    """
    executions = executions_from_state(state)
    question = state.get("question", "") or "(empty)"
    final_answer = state.get("final_answer") or ""
    correlation_id = state.get("correlation_id")
    metadata = dict(state.get("metadata") or {})

    retrieved_documents: list[RetrievedDocument] = []
    retrieved_chunks: list[str] = []
    prompt: dict[str, Any] | str | None = None
    model: str | None = None
    token_usage: dict[str, Any] = {}
    latency_total = sum(e.latency_ms or 0.0 for e in executions)

    rag_snapshot = metadata.get("last_rag_response")
    if isinstance(rag_snapshot, dict):
        prompt = (rag_snapshot.get("generation_metadata") or {}).get("prompt")
        model = (rag_snapshot.get("generation_metadata") or {}).get("model")
        gen_meta = rag_snapshot.get("generation_metadata") or {}
        token_usage = {
            key: gen_meta[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if key in gen_meta
        }
        for index, doc in enumerate(rag_snapshot.get("retrieved_documents") or []):
            if isinstance(doc, dict):
                text = str(doc.get("text") or "")
                if not text.strip():
                    continue
                retrieved_chunks.append(text)
                doc_id = str(doc.get("id") or f"chunk-{index}")
                try:
                    retrieved_documents.append(
                        RetrievedDocument(
                            id=doc_id,
                            text=text,
                            score=float(doc.get("score") or 0.0),
                            metadata=dict(doc.get("metadata") or {}),
                        )
                    )
                except Exception:  # noqa: BLE001 — keep finalize resilient
                    continue

    # Prefer summarize tool prompt/model when present.
    for execution in executions:
        if execution.tool_name == "summarize" and isinstance(execution.output, dict):
            prompt = execution.output.get("prompt", prompt)
            model_value = execution.output.get("model")
            if model_value is not None:
                model = str(model_value)

    context = EvaluationContext(
        question=question,
        original_user_request=question,
        retrieved_documents=retrieved_documents,
        retrieved_chunks=retrieved_chunks,
        prompt=prompt,
        tool_calls=executions,
        tool_results=[e.output for e in executions],
        answer=final_answer,
        model=str(model) if model is not None else None,
        latency_ms=latency_total,
        token_usage=token_usage,
        metadata={
            **{k: v for k, v in metadata.items() if k != "last_rag_response"},
            "selected_tools": list(state.get("selected_tools") or []),
            "tool_contracts": list(state.get("tool_contracts") or []),
            "planner_output": state.get("planner_output"),
        },
        langsmith_trace_id=state.get("trace_id"),
        langsmith_run_id=state.get("run_id"),
        langsmith_run_url=state.get("run_url"),
        correlation_id=correlation_id,
    )

    logger.info(
        "Finalize completed: correlation_id=%s tools=%s answer_len=%d",
        correlation_id,
        [e.tool_name for e in executions],
        len(final_answer),
    )
    return {
        "evaluation_context": context.model_dump(
            mode="json",
            exclude={"generated_answer"},
        ),
        "messages": [
            {
                "role": "assistant",
                "content": final_answer,
            }
        ],
    }


def build_agent_graph(
    *,
    planner: Planner,
    router: ToolRouter,
) -> Any:
    """Compile the Sprint 11 LangGraph agent.

    Graph::

        START → planner → router → finalize → END

    Args:
        planner: Planner node (callable).
        router: Tool router node (callable).

    Returns:
        Compiled LangGraph runnable.
    """
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("router", router)
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile()
    logger.info("LangGraph agent compiled: nodes=planner,router,finalize")
    return compiled


def build_default_graph(registry: ToolRegistry, planner: Planner | None = None) -> Any:
    """Convenience: compile graph with default planner + router.

    Args:
        registry: Tool registry.
        planner: Optional planner override.

    Returns:
        Compiled graph.
    """
    return build_agent_graph(
        planner=planner or Planner(),
        router=ToolRouter(registry),
    )


__all__ = [
    "build_agent_graph",
    "build_default_graph",
    "_finalize_node",
]
