"""ToolValidator — compare expected vs actual agent tool usage.

WHY custom logic (not DeepEval / LangSmith alone)
-------------------------------------------------
Vendor eval products score language quality and record traces. They do
not encode *your* agent contract: "must call ``search_docs`` before
``answer_user`` with ``{"query": ...}``". That is assertion logic —
closer to verifying a Selenium page-object interaction sequence than to
faithfulness scoring.

HOW this maps to future frameworks (Sprint 11)
----------------------------------------------
| Framework       | Mapper idea                                      |
|-----------------|--------------------------------------------------|
| LangGraph       | Read tool node events from graph state / checks  |
| OpenAI Agents   | Map ``tool_calls`` from response items           |
| CrewAI          | Map agent tool invocation logs                   |
| AutoGen         | Map function-call messages between agents        |
| MCP             | Map MCP ``tools/call`` request payloads          |

This sprint ships the validator only — no agent runtime.
"""

from __future__ import annotations

import logging
from typing import Any

from app.evaluation.tool_validation.comparator import (
    count_is_satisfied,
    filter_matching_calls,
    latency_is_satisfied,
    order_is_satisfied,
)
from app.evaluation.tool_validation.expectation import expectations_from_golden_fields
from app.evaluation.tool_validation.models import (
    ActualToolCall,
    ToolCallExpectation,
    ToolMatchResult,
)
from app.evaluation.tool_validation.report import ToolValidationReport
from app.evaluation.tool_validation.tool_contract import ToolContract
from app.evaluation.tool_validation.tool_execution import ToolExecution
from app.evaluation.tool_validation.trace_mapper import ToolTraceMapper

logger = logging.getLogger(__name__)


class ToolValidator:
    """Validate actual tool calls against golden expectations.

    Args:
        allow_extra_calls: When False, unexpected tool names fail the run.
        require_exact_arguments: Default exactness for golden-field helpers.
    """

    def __init__(
        self,
        *,
        allow_extra_calls: bool = True,
        require_exact_arguments: bool = False,
    ) -> None:
        self._allow_extra_calls = allow_extra_calls
        self._require_exact_arguments = require_exact_arguments

    def validate(
        self,
        expectations: list[ToolCallExpectation],
        actual_calls: list[ActualToolCall],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolValidationReport:
        """Run full validation and return a structured report.

        When ``expectations`` is empty, the result is a vacuously passing
        report (RAG-only examples with no tool contract).

        Args:
            expectations: Golden tool expectations.
            actual_calls: Observed tool calls (may be empty today).
            metadata: Optional report metadata.

        Returns:
            ``ToolValidationReport``.
        """
        matches: list[ToolMatchResult] = []
        failures: list[str] = []

        for expectation in expectations:
            matched = filter_matching_calls(expectation, actual_calls)
            item_failures: list[str] = []

            if not count_is_satisfied(expectation, len(matched)):
                item_failures.append(
                    f"Tool '{expectation.tool_name}' count={len(matched)} "
                    f"not in [{expectation.min_count}, {expectation.max_count}]"
                )
            if not order_is_satisfied(expectation, matched):
                item_failures.append(
                    f"Tool '{expectation.tool_name}' not found at order="
                    f"{expectation.order}"
                )
            if matched and not latency_is_satisfied(expectation, matched):
                item_failures.append(
                    f"Tool '{expectation.tool_name}' exceeded max_latency_ms="
                    f"{expectation.max_latency_ms}"
                )
            if not matched and expectation.min_count > 0:
                # Ensure a clear message when nothing matched at all.
                if not item_failures:
                    item_failures.append(
                        f"Expected tool '{expectation.tool_name}' was not called"
                    )

            passed = len(item_failures) == 0
            matches.append(
                ToolMatchResult(
                    expectation=expectation,
                    matched_calls=matched,
                    passed=passed,
                    failures=item_failures,
                )
            )
            failures.extend(item_failures)

        if not self._allow_extra_calls and expectations:
            expected_names = {e.tool_name for e in expectations}
            for call in actual_calls:
                if call.tool_name not in expected_names:
                    msg = f"Unexpected tool call: '{call.tool_name}'"
                    failures.append(msg)

        expected_tools = [e.tool_name for e in expectations]
        actual_tools = [c.tool_name for c in actual_calls]
        latency_total = sum(
            c.latency_ms for c in actual_calls if c.latency_ms is not None
        )
        report = ToolValidationReport(
            expected_tools=expected_tools,
            actual_tools=actual_tools,
            expectations=list(expectations),
            actual_calls=list(actual_calls),
            matches=matches,
            passed=len(failures) == 0,
            failures=failures,
            execution_count_expected=sum(e.min_count for e in expectations),
            execution_count_actual=len(actual_calls),
            latency_ms_total=float(latency_total),
            metadata=dict(metadata or {}),
        )
        logger.info(
            "Tool validation completed: passed=%s expected=%s actual=%s failures=%d",
            report.passed,
            expected_tools,
            actual_tools,
            len(failures),
        )
        return report

    def validate_from_golden(
        self,
        *,
        expected_tools: list[str] | None = None,
        expected_tool_arguments: list[dict[str, Any]] | None = None,
        expected_tool_order: list[str] | None = None,
        actual_calls: list[ActualToolCall] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolValidationReport:
        """Convenience API using golden-dataset field shapes.

        Args:
            expected_tools: Expected tool names.
            expected_tool_arguments: Aligned argument dicts.
            expected_tool_order: Ordered expected tool names.
            actual_calls: Observed calls (default empty — RAG-only today).
            metadata: Optional metadata.

        Returns:
            ``ToolValidationReport``.
        """
        expectations = expectations_from_golden_fields(
            expected_tools=expected_tools,
            expected_tool_arguments=expected_tool_arguments,
            expected_tool_order=expected_tool_order,
            require_exact_arguments=self._require_exact_arguments,
        )
        return self.validate(
            expectations,
            list(actual_calls or []),
            metadata=metadata,
        )

    def validate_contracts(
        self,
        contracts: list[ToolContract],
        executions: list[ToolExecution],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolValidationReport:
        """Validate ``ToolExecution`` records against declarative ``ToolContract``s.

        Bridges contracts → ``ToolCallExpectation`` and executions →
        ``ActualToolCall`` so the Sprint 10 comparator stays the single
        assertion engine. Also appends contract-level argument failures.

        Args:
            contracts: Expected tool behaviour specs.
            executions: Normalized tool executions (from ``ToolTraceMapper``).
            metadata: Optional report metadata.

        Returns:
            ``ToolValidationReport``.
        """
        mapper = ToolTraceMapper()
        actual_calls = mapper.to_actual_tool_calls(executions)
        expectations = [contract.to_expectation() for contract in contracts]
        report = self.validate(
            expectations,
            actual_calls,
            metadata=metadata,
        )

        extra_failures: list[str] = []
        by_name: dict[str, list[ToolExecution]] = {}
        for execution in executions:
            by_name.setdefault(execution.tool_name, []).append(execution)

        for contract in contracts:
            for execution in by_name.get(contract.tool_name, []):
                for failure in contract.validate_arguments(execution.arguments):
                    msg = f"[{contract.tool_name}] {failure}"
                    extra_failures.append(msg)
                if contract.expected_output_type and execution.output is not None:
                    actual_type = type(execution.output).__name__
                    expected_type = contract.expected_output_type
                    if actual_type != expected_type and expected_type not in {
                        "any",
                        "Any",
                    }:
                        # Accept common JSON aliases
                        aliases = {
                            "dict": {"dict"},
                            "list": {"list"},
                            "str": {"str"},
                            "int": {"int"},
                            "float": {"float", "int"},
                            "bool": {"bool"},
                        }
                        allowed = aliases.get(expected_type, {expected_type})
                        if actual_type not in allowed:
                            extra_failures.append(
                                f"[{contract.tool_name}] expected output type "
                                f"{expected_type}, got {actual_type}"
                            )

        if not extra_failures:
            return report

        all_failures = list(report.failures) + extra_failures
        return report.model_copy(
            update={
                "failures": all_failures,
                "passed": False,
            }
        )


__all__ = ["ToolValidator"]