"""ToolContract — declarative expected tool behaviour.

WHY contracts beat scattered assertions
---------------------------------------
Ad-hoc ``assert tool == "search_docs"`` checks in tests / notebooks:

* duplicate the same rules across suites
* drift from golden datasets
* cannot be versioned, reviewed, or reused by CI automation
* mix *policy* (what the agent must do) with *mechanism* (how we observed it)

A ``ToolContract`` is the **spec** for one tool in a scenario — like a
page-object contract or OpenAPI operation: required/optional args,
validators, call bounds, order, latency budget, and expected output type.
Golden datasets and ``ToolValidator`` consume contracts; Sprint 11 agents
only need to emit executions that satisfy them.

Java SDET analogy: a reusable assertion library for API contracts instead
of copy-pasted Hamcrest checks in every test class.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.tool_validation.models import ToolCallExpectation

ArgumentValidator = Callable[[Any], bool]


class ToolContract(BaseModel):
    """Expected behaviour for one tool in an evaluation scenario.

    Attributes:
        tool_name: Tool / function name.
        required: Argument names that must be present.
        optional: Argument names that may be present.
        expected_arguments: Concrete expected argument values
            (subset match unless converted with exact mode).
        argument_validators: Optional per-argument predicate map
            (not serialized; runtime-only).
        expected_execution_order: Optional 0-based sequence index.
        maximum_calls: Optional upper bound on invocations.
        minimum_calls: Lower bound on invocations (default 1).
        maximum_latency_ms: Optional per-call latency budget.
        expected_output_type: Optional expected Python / JSON type name
            (e.g. ``str``, ``dict``, ``list``).
        require_exact_arguments: Exact vs subset argument matching when
            bridged to ``ToolCallExpectation``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tool_name: str = Field(description="Tool name under contract")
    required: list[str] = Field(
        default_factory=list,
        description="Required argument names",
    )
    optional: list[str] = Field(
        default_factory=list,
        description="Optional argument names",
    )
    expected_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected argument values",
    )
    argument_validators: dict[str, ArgumentValidator] = Field(
        default_factory=dict,
        exclude=True,
        description="Runtime-only argument validators",
    )
    expected_execution_order: int | None = Field(
        default=None,
        ge=0,
        description="Optional expected index in the call sequence",
    )
    maximum_calls: int | None = Field(
        default=None,
        ge=0,
        description="Optional maximum invocation count",
    )
    minimum_calls: int = Field(
        default=1,
        ge=0,
        description="Minimum invocation count",
    )
    maximum_latency_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional max latency in milliseconds",
    )
    expected_output_type: str | None = Field(
        default=None,
        description="Optional expected output type name",
    )
    require_exact_arguments: bool = Field(
        default=False,
        description="Exact argument equality when bridged to expectations",
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_not_be_blank(cls, value: str) -> str:
        """Reject blank tool names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tool_name must be non-empty")
        return cleaned

    @field_validator("required", "optional")
    @classmethod
    def strip_arg_names(cls, value: list[str]) -> list[str]:
        """Normalize argument name lists."""
        return [item.strip() for item in value if item and item.strip()]

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Return human-readable failures for ``arguments`` against this contract.

        Checks required keys, rejects unknown keys outside required∪optional
        when either list is non-empty, runs argument validators, and compares
        ``expected_arguments`` as a subset (or exact when configured).

        Args:
            arguments: Actual tool arguments.

        Returns:
            Failure messages (empty when valid).
        """
        failures: list[str] = []
        for name in self.required:
            if name not in arguments:
                failures.append(f"Missing required argument '{name}'")

        allowed = set(self.required) | set(self.optional)
        if allowed:
            for key in arguments:
                if key not in allowed and key not in self.expected_arguments:
                    failures.append(f"Unexpected argument '{key}'")

        for key, expected_value in self.expected_arguments.items():
            if key not in arguments:
                failures.append(f"Missing expected argument '{key}'")
            elif arguments[key] != expected_value:
                failures.append(
                    f"Argument '{key}' expected {expected_value!r}, "
                    f"got {arguments[key]!r}"
                )

        if self.require_exact_arguments and self.expected_arguments != arguments:
            failures.append("Arguments do not match exactly")

        for key, validator in self.argument_validators.items():
            if key not in arguments:
                failures.append(f"Validator target '{key}' missing")
                continue
            try:
                ok = bool(validator(arguments[key]))
            except Exception as exc:  # noqa: BLE001 — surface validator crashes
                failures.append(f"Validator for '{key}' raised: {exc}")
                continue
            if not ok:
                failures.append(f"Argument '{key}' failed custom validator")

        return failures

    def to_expectation(self) -> ToolCallExpectation:
        """Bridge to the Sprint 10 ``ToolCallExpectation`` used by ``ToolValidator``.

        Returns:
            Equivalent ``ToolCallExpectation``.
        """
        return ToolCallExpectation(
            tool_name=self.tool_name,
            arguments=dict(self.expected_arguments),
            order=self.expected_execution_order,
            min_count=self.minimum_calls,
            max_count=self.maximum_calls,
            max_latency_ms=self.maximum_latency_ms,
            require_exact_arguments=self.require_exact_arguments,
        )

    @classmethod
    def from_golden_fields(
        cls,
        *,
        tool_name: str,
        expected_arguments: dict[str, Any] | None = None,
        expected_execution_order: int | None = None,
        required: list[str] | None = None,
        optional: list[str] | None = None,
        minimum_calls: int = 1,
        maximum_calls: int | None = None,
        maximum_latency_ms: float | None = None,
        expected_output_type: str | None = None,
        require_exact_arguments: bool = False,
    ) -> ToolContract:
        """Build a contract from golden-dataset style fields.

        Args:
            tool_name: Tool name.
            expected_arguments: Expected argument map.
            expected_execution_order: Optional sequence index.
            required: Required argument names (defaults to expected keys).
            optional: Optional argument names.
            minimum_calls: Min invocations.
            maximum_calls: Max invocations.
            maximum_latency_ms: Latency budget.
            expected_output_type: Output type name.
            require_exact_arguments: Exact arg matching flag.

        Returns:
            ``ToolContract``.
        """
        args = dict(expected_arguments or {})
        return cls(
            tool_name=tool_name,
            required=list(required) if required is not None else list(args.keys()),
            optional=list(optional or []),
            expected_arguments=args,
            expected_execution_order=expected_execution_order,
            minimum_calls=minimum_calls,
            maximum_calls=maximum_calls,
            maximum_latency_ms=maximum_latency_ms,
            expected_output_type=expected_output_type,
            require_exact_arguments=require_exact_arguments,
        )


def contracts_from_golden_fields(
    *,
    expected_tools: list[str] | None = None,
    expected_tool_arguments: list[dict[str, Any]] | None = None,
    expected_tool_order: list[str] | None = None,
    require_exact_arguments: bool = False,
) -> list[ToolContract]:
    """Build contracts from golden dataset columns.

    Mirrors ``expectations_from_golden_fields`` but returns ``ToolContract``
    objects so scenarios can share one declarative source of truth.

    Args:
        expected_tools: Expected tool names.
        expected_tool_arguments: Per-tool argument dicts aligned by index.
        expected_tool_order: Ordered expected tool names.
        require_exact_arguments: Exact vs subset argument matching.

    Returns:
        List of ``ToolContract``.
    """
    args_list = list(expected_tool_arguments or [])
    names = list(expected_tool_order) if expected_tool_order else list(expected_tools or [])
    use_order = bool(expected_tool_order)
    contracts: list[ToolContract] = []
    for index, name in enumerate(names):
        arguments = args_list[index] if index < len(args_list) else {}
        contracts.append(
            ToolContract.from_golden_fields(
                tool_name=name,
                expected_arguments=dict(arguments),
                expected_execution_order=index if use_order else None,
                require_exact_arguments=require_exact_arguments,
            )
        )
    return contracts


__all__ = [
    "ArgumentValidator",
    "ToolContract",
    "contracts_from_golden_fields",
]
