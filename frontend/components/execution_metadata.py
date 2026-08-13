"""Extract / sanitize agent execution metadata for Developer Mode.

Presentation-only helpers. No routing or agent logic lives here.
"""

from __future__ import annotations

from typing import Any

_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "bearer",
    "openai",
    "langsmith_api",
)

_SENSITIVE_EXACT = {
    "token",
    "password",
    "secret",
    "authorization",
}

_EMPLOYEE_ROUTES = frozenset(
    {"employee", "hybrid", "verify_id", "EMPLOYEE_REQUEST", "employee_request"}
)


def sanitize_for_developer_view(value: Any) -> Any:
    """Redact credential-like keys and trim oversized strings."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _SENSITIVE_EXACT or any(
                fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS
            ):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = sanitize_for_developer_view(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_for_developer_view(item) for item in value[:30]]
    if isinstance(value, str) and len(value) > 800:
        return value[:800] + "…"
    return value


def _agent_payload(
    payload: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if mode == "DEMO":
        agent = payload.get("agent_result")
        return agent if isinstance(agent, dict) else None
    if mode == "AGENT":
        return payload
    return None


def _source_label(doc: Any) -> str | None:
    if isinstance(doc, dict):
        doc_meta = doc.get("metadata") or {}
        source = (
            doc_meta.get("source")
            or doc_meta.get("file_name")
            or doc_meta.get("title")
            or doc.get("id")
        )
        return str(source) if source else None
    if isinstance(doc, str) and doc.strip():
        return doc.strip()
    return None


def _score_from_doc(doc: Any) -> float | None:
    if not isinstance(doc, dict):
        return None
    score = doc.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def extract_execution_metadata(
    payload: dict[str, Any] | None,
    mode: str,
    *,
    verify_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Developer Mode–safe snapshot of execution metadata."""
    agent = _agent_payload(payload, mode)
    meta = sanitize_for_developer_view((agent or {}).get("metadata") or {})
    if not isinstance(meta, dict):
        meta = {}

    planner = (agent or {}).get("planner_output") or (agent or {}).get(
        "planner_decision"
    ) or {}
    tools = (agent or {}).get("tool_executions") or meta.get("tools_invoked") or []
    ctx = (agent or {}).get("evaluation_context") or {}

    retrieved_docs = list(ctx.get("retrieved_documents") or [])
    retrieved_chunks = list(ctx.get("retrieved_chunks") or [])
    rag_snapshot = meta.get("last_rag_response")
    if isinstance(rag_snapshot, dict) and not retrieved_docs:
        retrieved_docs = list(rag_snapshot.get("retrieved_documents") or [])

    sources: list[str] = []
    scores: list[float] = []
    for doc in retrieved_docs[:10]:
        label = _source_label(doc)
        if label:
            sources.append(label)
        score = _score_from_doc(doc)
        if score is not None:
            scores.append(score)

    rag_used = meta.get("rag_used")
    if rag_used is None:
        rag_used = bool(retrieved_docs or retrieved_chunks) or any(
            (t.get("tool_name") if isinstance(t, dict) else None)
            in {"search_docs", "summarize"}
            for t in tools
        )

    verification = meta.get("verification_status")
    verified_id = meta.get("verified_employee_id")
    if verify_result and verify_result.get("verified"):
        verification = "verified"
        verified_id = verify_result.get("employee_id") or verified_id
    elif verification is None:
        verification = "verified" if verified_id else "unverified"

    tool_rows: list[dict[str, Any]] = []
    if isinstance(meta.get("tools_invoked"), list) and meta["tools_invoked"]:
        for item in meta["tools_invoked"]:
            if isinstance(item, dict):
                tool_rows.append(sanitize_for_developer_view(item))
    else:
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            status = tool.get("status", "—")
            if isinstance(status, dict):
                status = status.get("value", status)
            tool_rows.append(
                sanitize_for_developer_view(
                    {
                        "tool_name": tool.get("tool_name"),
                        "status": status,
                        "arguments": tool.get("arguments") or {},
                        "result_summary": _brief_output(tool.get("output")),
                        "latency_ms": tool.get("latency_ms"),
                        "error": "tool_error" if tool.get("error") else None,
                    }
                )
            )

    intent = (
        meta.get("detected_intent")
        or meta.get("intent_route")
        or planner.get("intent_route")
    )
    route = meta.get("selected_route") or intent

    run_url = (
        (agent or {}).get("run_url")
        or meta.get("langsmith_run_url")
        or ctx.get("langsmith_run_url")
    )
    latency = (agent or {}).get("latency_ms")
    if latency is None:
        latency = meta.get("latency_ms")

    # Derive verification_required from existing route / tool signals only.
    intent_key = str(intent or "").strip().lower()
    verification_required = intent_key in _EMPLOYEE_ROUTES or any(
        str((t.get("tool_name") if isinstance(t, dict) else "") or "").startswith(
            "get_"
        )
        or str((t.get("tool_name") if isinstance(t, dict) else "") or "")
        == "verify_employee"
        for t in (tool_rows or tools)
    )

    top_k: Any = None
    retrieval_query: Any = None
    model_name: Any = ctx.get("model") if isinstance(ctx, dict) else None
    if isinstance(rag_snapshot, dict):
        retrieval_meta = rag_snapshot.get("retrieval_metadata") or {}
        if isinstance(retrieval_meta, dict):
            top_k = (
                retrieval_meta.get("top_k")
                or retrieval_meta.get("requested_top_k")
                or retrieval_meta.get("retrieved_count")
            )
            retrieval_query = retrieval_meta.get("query") or retrieval_meta.get(
                "rewritten_query"
            )
        gen_meta = rag_snapshot.get("generation_metadata") or {}
        if isinstance(gen_meta, dict) and not model_name:
            model_name = gen_meta.get("model")
        if top_k is None and retrieved_docs:
            top_k = len(retrieved_docs)

    if mode == "RAG" and isinstance(payload, dict):
        docs = payload.get("retrieved_documents") or []
        latency_block = payload.get("latency") or {}
        gen = payload.get("generation_metadata") or {}
        retrieval_meta = payload.get("retrieval_metadata") or {}
        rag_scores = [
            score
            for score in (_score_from_doc(doc) for doc in docs)
            if score is not None
        ]
        return sanitize_for_developer_view(
            {
                "original_input": payload.get("question"),
                "normalized_input": payload.get("question"),
                "detected_intent": "knowledge",
                "selected_route": "rag",
                "verification_required": False,
                "verification_status": "not_required",
                "verified_employee_id": verified_id,
                "tools_invoked": [],
                "rag_used": True,
                "retrieved_sources": [
                    label
                    for label in (_source_label(d) for d in docs if isinstance(d, dict))
                    if label
                ],
                "retrieved_document_count": len(docs),
                "retrieval_scores": rag_scores,
                "top_k": (
                    retrieval_meta.get("top_k")
                    or retrieval_meta.get("retrieved_count")
                    or len(docs)
                    or None
                ),
                "retrieval_query": retrieval_meta.get("query"),
                "latency_ms": latency_block.get("total_ms"),
                "model": gen.get("model"),
                "langsmith_run_url": None,
            }
        )

    original = (agent or {}).get("question")
    normalized = meta.get("normalized_input") or original
    routing_normalized = meta.get("routing_normalized_input")

    return sanitize_for_developer_view(
        {
            "original_input": original,
            "normalized_input": normalized,
            "routing_normalized_input": routing_normalized,
            "detected_intent": intent,
            "selected_route": route,
            "verification_required": bool(verification_required),
            "verification_status": (
                "not_required"
                if not verification_required and verification == "unverified"
                else verification
            ),
            "verified_employee_id": verified_id,
            "tools_invoked": tool_rows,
            "rag_used": bool(rag_used),
            "retrieved_sources": sources,
            "retrieved_document_count": len(retrieved_docs) or len(retrieved_chunks),
            "retrieval_scores": scores,
            "top_k": top_k,
            "retrieval_query": retrieval_query,
            "latency_ms": latency,
            "model": model_name,
            "langsmith_run_url": run_url,
            "correlation_id": (agent or {}).get("correlation_id"),
            "trace_id": (agent or {}).get("trace_id") or ctx.get("langsmith_trace_id"),
            "tool_mode": meta.get("tool_mode"),
            "mcp": meta.get("mcp") if isinstance(meta.get("mcp"), dict) else None,
        }
    )


def _brief_output(output: Any) -> str:
    if output is None:
        return "empty"
    if isinstance(output, str):
        return output[:160] + ("…" if len(output) > 160 else "")
    if isinstance(output, dict):
        if "leave_history" in output:
            return f"leave_history entries={len(output.get('leave_history') or [])}"
        if "leave_balance" in output:
            lb = output.get("leave_balance") or {}
            return (
                "leave_balance "
                f"vacation={lb.get('vacation')} sick={lb.get('sick')} "
                f"personal={lb.get('personal')}"
            )
        keys = ", ".join(list(output.keys())[:5])
        return f"dict keys=[{keys}]"
    if isinstance(output, list):
        return f"list length={len(output)}"
    return type(output).__name__
