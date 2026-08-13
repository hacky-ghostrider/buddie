"""ToolTraceMapper — LangSmith (and similar) traces → ToolExecution.

WHY isolate the mapper
----------------------
``ToolValidator`` must stay vendor-blind. If it imported LangSmith run
trees, swapping to OpenTelemetry / Phoenix / a LangGraph checkpointer
dump would rewrite assertion code. The mapper is the **Anti-Corruption
Layer**: vendor JSON in, ``ToolExecution`` out, then
``ActualToolCall`` for the existing validator.

Pipeline:

```text
LangSmith Trace
      ↓
ToolTraceMapper
      ↓
ToolExecution[]
      ↓
ActualToolCall[]  →  ToolValidator
```

Sprint 10.2 ships the mapper with **no live LangSmith calls**. Tests and
dry-runs feed dict payloads that resemble LangSmith child runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from app.evaluation.tool_validation.models import ActualToolCall
from app.evaluation.tool_validation.tool_execution import (
    ToolExecution,
    ToolExecutionStatus,
)


class ToolTraceMapper:
    """Convert vendor trace payloads into internal tool execution models.

    Accepts loose mappings (dicts) so unit tests can mock LangSmith without
    the SDK. Recognized shapes:

    * ``{"tool_calls": [ ... ]}``
    * ``{"child_runs": [ ... ]}`` / ``{"runs": [ ... ]}``
    * a bare list of tool-call-like dicts
    * objects with ``tool_calls`` / ``child_runs`` attributes
    """

    def map_langsmith_trace(self, trace: Any) -> list[ToolExecution]:
        """Map a LangSmith-like trace payload to ``ToolExecution`` records.

        Args:
            trace: Dict, list, or object exposing tool / child run data.

        Returns:
            Ordered ``ToolExecution`` list (may be empty).
        """
        raw_calls = self._extract_raw_calls(trace)
        executions: list[ToolExecution] = []
        for index, raw in enumerate(raw_calls):
            if not isinstance(raw, Mapping):
                continue
            executions.append(self._map_one(dict(raw), order=index))
        return executions

    def to_actual_tool_calls(
        self,
        executions: Sequence[ToolExecution],
    ) -> list[ActualToolCall]:
        """Bridge internal executions to Sprint 10 ``ActualToolCall`` models.

        Args:
            executions: Normalized tool executions.

        Returns:
            ``ActualToolCall`` list for ``ToolValidator``.
        """
        calls: list[ActualToolCall] = []
        for execution in executions:
            calls.append(
                ActualToolCall(
                    tool_name=execution.tool_name,
                    arguments=dict(execution.arguments),
                    order=execution.order,
                    latency_ms=execution.latency_ms,
                    success=execution.success,
                    metadata={
                        **dict(execution.trace_metadata),
                        "status": (
                            execution.status.value
                            if hasattr(execution.status, "value")
                            else execution.status
                        ),
                        "retry_count": execution.retry_count,
                        "error": execution.error,
                    },
                )
            )
        return calls

    def map_to_actual_tool_calls(self, trace: Any) -> list[ActualToolCall]:
        """Convenience: LangSmith-like trace → ``ActualToolCall`` list.

        Args:
            trace: Vendor trace payload.

        Returns:
            Validator-ready actual calls.
        """
        return self.to_actual_tool_calls(self.map_langsmith_trace(trace))

    def _map_one(self, raw: dict[str, Any], *, order: int) -> ToolExecution:
        """Map one raw tool-call dict into ``ToolExecution``."""
        name = (
            raw.get("tool_name")
            or raw.get("name")
            or raw.get("tool")
            or ((raw.get("extra") or {}).get("metadata") or {}).get("tool_name")
        )
        if not name:
            name = "unknown_tool"

        arguments = raw.get("arguments") or raw.get("inputs") or raw.get("args") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}

        output = raw.get("output")
        if output is None:
            output = raw.get("outputs")

        status = self._coerce_status(raw)
        error = raw.get("error") or raw.get("error_message")
        if error is not None:
            error = str(error)

        latency = raw.get("latency_ms")
        if latency is None and raw.get("latency") is not None:
            latency = raw.get("latency")
        if latency is None:
            latency = self._latency_from_timestamps(raw)

        started_at = self._parse_dt(raw.get("started_at") or raw.get("start_time"))
        finished_at = self._parse_dt(raw.get("finished_at") or raw.get("end_time"))

        retry_count = int(raw.get("retry_count") or raw.get("retries") or 0)
        trace_metadata = {
            key: raw[key]
            for key in ("id", "run_id", "trace_id", "parent_run_id", "run_type")
            if key in raw
        }
        if "metadata" in raw and isinstance(raw["metadata"], dict):
            trace_metadata.update(raw["metadata"])

        return ToolExecution(
            tool_name=str(name),
            arguments=dict(arguments),
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            latency_ms=float(latency) if latency is not None else None,
            status=status,
            error=error,
            retry_count=max(0, retry_count),
            trace_metadata=trace_metadata,
            order=int(raw.get("order", order)),
        )

    @staticmethod
    def _extract_raw_calls(trace: Any) -> list[Any]:
        """Pull a list of tool-call-like dicts from varied trace shapes."""
        if trace is None:
            return []
        if isinstance(trace, list):
            return list(trace)
        if isinstance(trace, Mapping):
            for key in ("tool_calls", "child_runs", "runs", "tools"):
                value = trace.get(key)
                if isinstance(value, list):
                    return list(value)
            # Single tool-shaped mapping
            if any(k in trace for k in ("tool_name", "name", "arguments", "inputs")):
                return [trace]
            return []
        for attr in ("tool_calls", "child_runs", "runs"):
            value = getattr(trace, attr, None)
            if isinstance(value, list):
                return list(value)
        return []

    @staticmethod
    def _coerce_status(raw: dict[str, Any]) -> ToolExecutionStatus:
        """Derive a coarse status from vendor fields."""
        explicit = raw.get("status")
        if explicit is not None:
            return ToolExecutionStatus.coerce(explicit)
        if raw.get("error") or raw.get("error_message"):
            return ToolExecutionStatus.FAILED
        if raw.get("success") is False:
            return ToolExecutionStatus.FAILED
        if raw.get("success") is True:
            return ToolExecutionStatus.SUCCESS
        # LangSmith often uses run_type=tool with outputs present
        if raw.get("output") is not None or raw.get("outputs") is not None:
            return ToolExecutionStatus.SUCCESS
        return ToolExecutionStatus.FAILED

    @staticmethod
    def _latency_from_timestamps(raw: dict[str, Any]) -> float | None:
        """Compute latency from start/end timestamps when present."""
        start = ToolTraceMapper._parse_dt(
            raw.get("started_at") or raw.get("start_time")
        )
        end = ToolTraceMapper._parse_dt(raw.get("finished_at") or raw.get("end_time"))
        if start is None or end is None:
            return None
        return max(0.0, (end - start).total_seconds() * 1000.0)

    @staticmethod
    def _parse_dt(value: Any) -> datetime | None:
        """Parse datetime from datetime / ISO string / None."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None


__all__ = ["ToolTraceMapper"]
