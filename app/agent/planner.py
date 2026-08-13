"""Planner node — decide required tools, order, and ToolContracts.

The planner is the *architect*: it does not execute tools. It emits a
structured ``PlannerOutput`` that the router consumes. For production
determinism and offline tests, the default strategy is rule-based; an
optional LLM planner protocol can be injected later without changing the
graph.

Conversational / unsupported intents are classified *before* RAG so
greetings never trigger retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from app.agent.conversation import (
    IntentRoute,
    UNSUPPORTED_FALLBACK,
    VERIFY_PROMPT,
    classify_intent,
    is_knowledge_question,
)
from app.agent.exceptions import AgentPlanningError
from app.agent.models import PlannerOutput, ToolInvocation
from app.agent.state import AgentState
from app.evaluation.scenarios import CANONICAL_SCENARIO_ID
from app.evaluation.tool_validation.tool_contract import ToolContract

logger = logging.getLogger(__name__)

_CANONICAL_LEAVE_HINTS = (
    "leave policy",
    "employee handbook",
    "summarize the leave",
)

_HYBRID_POLICY_HINTS = (
    "carry forward",
    "carry-forward",
    "rollover",
    "roll over",
    "can i carry",
    "remaining vacation days",
)

_LEAVE_BALANCE_HINTS = (
    "leave balance",
    "vacation days",
    "pto left",
    "days do i have",
    "days left",
    "how many vacation",
    "sick leave do i",
    "personal leave do i",
    "time off i have",
    # Natural balance phrasings (incl. conversational follow-ups).
    "how many leaves",
    "leaves do i",
    "leaves are there",
    "leaves are remaining",
    "remaining leaves",
    "my leaves",
    "my leave",
    "how much pto",
    "pto do i",
    "how many pto",
    "days off",
    "days off can i",
    "time off do i",
    "vacation do i",
    "sick days do i",
)

# Quantity / remaining signals that, with a leave-stock term, mean balance.
_LEAVE_BALANCE_QUANTITY_HINTS = (
    "how many",
    "how much",
    "do i have",
    "are there",
    "remaining",
    "left",
    "balance",
    "available",
)

_LEAVE_STOCK_TERMS = (
    "leaves",
    "leave",
    "pto",
    "vacation",
    "time off",
    "days off",
    "sick days",
    "personal days",
)

_LEAVE_HISTORY_HINTS = (
    "leave history",
    "did i take",
    "days did i take",
    "leave usage",
    "vacation days did",
    "sick leave did",
    "historically",
    "last year",
    "in 2024",
    "in 2025",
    "versus 2025",
    "vs 2025",
)

_UPCOMING_LEAVE_HINTS = (
    "next vacation",
    "upcoming leave",
    "pending leave",
    "leave request",
    "when is my next",
)

_PENDING_ACTION_HINTS = (
    "pending action",
    "pending task",
    "hr task",
    "due this week",
    "anything due",
    "pending for me",
)

_PAYROLL_HINTS = (
    "payroll",
    "my salary",
    "pay date",
    "paycheck",
    "net pay",
    "gross pay",
)

_ATTENDANCE_HINTS = (
    "attendance",
    "days was i absent",
    "days absent",
    "late days",
    "days present",
)

_HOLIDAY_HINTS = (
    "company holiday",
    "upcoming holiday",
    "holidays coming",
    "holiday next",
    "next holiday",
    "holidays are coming",
    "holiday calendar",
    "what holidays",
)

_PROFILE_HINTS = (
    "my department",
    "my profile",
    "my designation",
    "where do i work",
    "my location",
)

_MANAGER_HINTS = (
    "my manager",
    "who is my manager",
    "who manages me",
    "manager name",
)

_LEAVE_ELIGIBILITY_HINTS = (
    "can i take",
    "am i eligible",
    "eligible for",
    "do i have enough",
    "enough vacation",
    "enough sick",
    "enough personal",
    "may i take",
    "could i take",
)

_LEAVE_REQUEST_HINTS = (
    "i want to take",
    "i'd like to take",
    "i would like to take",
    "request leave",
    "apply for leave",
    "book vacation",
    "submit a leave",
    "create a leave",
    "take leave",
    "take vacation",
)

_CONFIRM_HINTS = (
    "confirm",
    "yes confirm",
    "confirm please",
    "go ahead",
    "submit it",
    "submit the request",
    "create it",
    "create the request",
    "yes, create",
    "yes create",
    "please proceed",
    "proceed",
)

_CANCEL_HINTS = (
    "cancel",
    "don't create",
    "do not create",
    "don't submit",
    "do not submit",
    "never mind",
    "nevermind",
)


class PlannerStrategy(Protocol):
    """Strategy interface for producing ``PlannerOutput``."""

    def plan(self, question: str, *, metadata: dict[str, Any] | None = None) -> PlannerOutput:
        """Create a structured plan for ``question``."""


class RuleBasedPlanner:
    """Deterministic planner covering demo + employee + RAG intents.

    Rules (first match wins):
        0. Conversational / empty / unknown / unsupported → direct answer
           (no RAG, no employee tools).
        1. Standalone employee id → ``verify_employee``.
        2. Canonical leave-policy / handbook summarize → ``search_docs`` then
           ``summarize`` (matches ``agent-tools-foundation-001``).
        3. Hybrid leave-balance + policy (carry-forward) → employee + RAG.
        4. Employee structured-data intents → protected employee tools
           (verification gate when session is unverified).
        5. Company holidays → ``get_upcoming_holidays``.
        6. Arithmetic expression → ``calculator``.
        7. Explicit web-search intent → ``search``.
        8. Enterprise knowledge / policy → ``search_docs`` then ``summarize``.
        9. Otherwise → graceful unsupported fallback (never blind RAG).
    """

    def plan(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        """Return a structured plan with contracts."""
        cleaned = (question or "").strip()
        meta = metadata or {}
        scenario = str(meta.get("scenario") or meta.get("golden_id") or "")
        verified = bool(meta.get("employee_id") or meta.get("verified_employee_id"))

        intent = classify_intent(cleaned, metadata=meta)
        # Human-in-the-loop confirmation must win over conversational acks.
        pending = meta.get("pending_leave_request")
        if isinstance(pending, dict) and pending:
            if self._is_leave_confirmation(cleaned):
                return self._plan_confirmed_leave_request(pending, verified=verified)
            if self._is_leave_cancellation(cleaned):
                return self._plan_direct(
                    "Okay — I cancelled the pending leave request. "
                    "Nothing was submitted.",
                    route=IntentRoute.EMPLOYEE.value,
                    rationale="User cancelled pending leave write action.",
                )

        if intent.route in {
            IntentRoute.EMPTY,
            IntentRoute.CONVERSATION,
            IntentRoute.UNKNOWN,
            IntentRoute.UNSUPPORTED,
        }:
            logger.info(
                "Planner conversational route: route=%s kind=%s",
                intent.route.value,
                getattr(intent.kind, "value", None),
            )
            return self._plan_direct(
                intent.response or "",
                route=intent.route.value,
                rationale=f"Conversational route: {intent.route.value}.",
            )

        if intent.route == IntentRoute.VERIFY_ID and intent.employee_id:
            return self._plan_verify_id(intent.employee_id)

        if not cleaned:
            raise AgentPlanningError("Cannot plan for an empty question")

        if scenario == CANONICAL_SCENARIO_ID or self._is_leave_policy_question(cleaned):
            return self._plan_search_docs_summarize(
                cleaned,
                query="leave policy employee handbook",
                document="employee_handbook.pdf",
                rationale=(
                    "Document policy question requires retrieve-then-summarize "
                    f"(scenario={scenario or 'leave-policy-heuristic'})."
                ),
                intent_route=IntentRoute.KNOWLEDGE.value,
            )

        if self._is_hybrid_leave_policy(cleaned):
            if not verified:
                return self._plan_direct(
                    VERIFY_PROMPT,
                    route=IntentRoute.EMPLOYEE.value,
                    rationale="Hybrid leave question requires verification first.",
                )
            return self._plan_hybrid_leave_and_policy()

        # Multi-tool: manager + holidays (independent reads).
        if self._looks_like_manager_and_holidays(cleaned):
            if not verified:
                return self._plan_direct(
                    VERIFY_PROMPT,
                    route=IntentRoute.EMPLOYEE.value,
                    rationale="Manager + holidays requires verification first.",
                )
            return self._plan_manager_and_holidays(cleaned)

        # Multi-tool: leave request (eligibility then human confirmation).
        if self._looks_like_leave_request(cleaned):
            if not verified:
                return self._plan_direct(
                    VERIFY_PROMPT,
                    route=IntentRoute.EMPLOYEE.value,
                    rationale="Leave request requires verification first.",
                )
            return self._plan_leave_request_draft(cleaned)

        # Multi-tool: leave eligibility check.
        if self._looks_like_leave_eligibility(cleaned):
            if not verified:
                return self._plan_direct(
                    VERIFY_PROMPT,
                    route=IntentRoute.EMPLOYEE.value,
                    rationale="Leave eligibility requires verification first.",
                )
            return self._plan_leave_eligibility(cleaned)

        employee_plan = self._plan_employee_intent(cleaned, verified=verified)
        if employee_plan is not None:
            return employee_plan

        if self._looks_like_holidays(cleaned):
            country, year = self._extract_country_year(cleaned)
            if country or year:
                args: dict[str, Any] = {}
                if country:
                    args["country"] = country
                if year:
                    args["year"] = year
                return self._plan_single_tool(
                    "get_holiday_calendar",
                    arguments=args,
                    rationale="Holiday calendar by country/year.",
                    intent_route=IntentRoute.EMPLOYEE.value,
                )
            return self._plan_single_tool(
                "get_upcoming_holidays",
                arguments={"limit": 5},
                rationale="Company holiday calendar question.",
                intent_route=IntentRoute.EMPLOYEE.value,
            )

        if self._looks_like_math(cleaned):
            return PlannerOutput(
                required_tools=["calculator"],
                optional_tools=[],
                alternative_tools=[],
                execution_order=["calculator"],
                invocations=[
                    ToolInvocation(
                        tool_name="calculator",
                        arguments={"expression": cleaned},
                        order=0,
                    )
                ],
                tool_contracts=[
                    ToolContract.from_golden_fields(
                        tool_name="calculator",
                        expected_arguments={"expression": cleaned},
                        expected_execution_order=0,
                        required=["expression"],
                        maximum_latency_ms=5_000.0,
                        expected_output_type="dict",
                    )
                ],
                rationale="Question looks like an arithmetic expression.",
                intent_route="calculator",
            )

        if self._looks_like_web_search(cleaned):
            query = cleaned
            return PlannerOutput(
                required_tools=["search"],
                optional_tools=[],
                alternative_tools=["search_docs"],
                execution_order=["search"],
                invocations=[
                    ToolInvocation(
                        tool_name="search",
                        arguments={"query": query},
                        order=0,
                    )
                ],
                tool_contracts=[
                    ToolContract.from_golden_fields(
                        tool_name="search",
                        expected_arguments={"query": query},
                        expected_execution_order=0,
                        required=["query"],
                        expected_output_type="dict",
                    )
                ],
                rationale="Question asks for external/web search.",
                intent_route="search",
            )

        if is_knowledge_question(cleaned):
            return self._plan_search_docs_summarize(
                cleaned,
                query=cleaned,
                document="knowledge_base",
                rationale="Enterprise knowledge / policy question → RAG.",
                intent_route=IntentRoute.KNOWLEDGE.value,
            )

        # Never blind-retrieve for unrelated / ambiguous chatter.
        logger.info("Planner unsupported/unknown fallback: preview=%r", cleaned[:80])
        return self._plan_direct(
            UNSUPPORTED_FALLBACK,
            route=IntentRoute.UNSUPPORTED.value,
            rationale="Outside supported employee domain; skip RAG.",
        )

    def _plan_employee_intent(
        self,
        question: str,
        *,
        verified: bool,
    ) -> PlannerOutput | None:
        lowered = question.lower()

        if self._contains_any(lowered, _PAYROLL_HINTS):
            return self._plan_protected("get_payroll_summary", verified=verified)
        if self._contains_any(lowered, _PENDING_ACTION_HINTS):
            return self._plan_protected("get_pending_actions", verified=verified)
        if self._contains_any(lowered, _UPCOMING_LEAVE_HINTS):
            return self._plan_protected("get_upcoming_leave", verified=verified)
        if self._contains_any(lowered, _LEAVE_HISTORY_HINTS):
            year = self._extract_year(lowered)
            args: dict[str, Any] = {}
            if year is not None:
                args["year"] = year
            return self._plan_protected(
                "get_leave_history",
                verified=verified,
                arguments=args,
            )
        if self._contains_any(lowered, _ATTENDANCE_HINTS):
            return self._plan_protected("get_attendance_summary", verified=verified)
        if self._looks_like_leave_balance(lowered):
            return self._plan_protected("get_leave_balance", verified=verified)
        if self._contains_any(lowered, _MANAGER_HINTS):
            return self._plan_protected("get_manager_information", verified=verified)
        if self._contains_any(lowered, _PROFILE_HINTS):
            return self._plan_protected("get_employee_profile", verified=verified)
        return None

    def _plan_hybrid_leave_and_policy(self) -> PlannerOutput:
        """Employee leave balance + explicit company policy retrieval."""
        tools: list[str] = ["get_leave_balance", "search_company_policy"]
        invocations: list[ToolInvocation] = [
            ToolInvocation(
                tool_name="get_leave_balance",
                arguments={},
                order=0,
            ),
            ToolInvocation(
                tool_name="search_company_policy",
                arguments={"query": "vacation carry forward leave policy"},
                order=1,
            ),
        ]
        contracts: list[ToolContract] = [
            ToolContract.from_golden_fields(
                tool_name="get_leave_balance",
                expected_arguments={},
                expected_execution_order=0,
                required=[],
                expected_output_type="dict",
            ),
            ToolContract.from_golden_fields(
                tool_name="search_company_policy",
                expected_arguments={"query": "vacation carry forward leave policy"},
                expected_execution_order=1,
                required=["query"],
                expected_output_type="dict",
            ),
        ]

        return PlannerOutput(
            required_tools=tools,
            optional_tools=[],
            alternative_tools=[],
            execution_order=tools,
            invocations=invocations,
            tool_contracts=contracts,
            rationale=(
                "Hybrid workflow: verified leave balance + handbook "
                "carry-forward policy via search_company_policy."
            ),
            intent_route=IntentRoute.HYBRID.value,
        )

    def _plan_leave_eligibility(self, question: str) -> PlannerOutput:
        leave_type, days = self._extract_leave_request_params(question)
        include_policy = "policy" in question.lower() or days >= 10
        tools = [
            "get_employee_profile",
            "get_leave_balance",
            "check_leave_eligibility",
        ]
        invocations: list[ToolInvocation] = [
            ToolInvocation(tool_name="get_employee_profile", arguments={}, order=0),
            ToolInvocation(tool_name="get_leave_balance", arguments={}, order=1),
            ToolInvocation(
                tool_name="check_leave_eligibility",
                arguments={
                    "leave_type": leave_type,
                    "requested_days": days,
                },
                order=2,
            ),
        ]
        if include_policy:
            tools.append("search_company_policy")
            invocations.append(
                ToolInvocation(
                    tool_name="search_company_policy",
                    arguments={
                        "query": f"{leave_type.lower()} leave eligibility policy"
                    },
                    order=3,
                )
            )
        contracts = [
            ToolContract.from_golden_fields(
                tool_name=inv.tool_name,
                expected_arguments=dict(inv.arguments),
                expected_execution_order=inv.order,
                required=list(inv.arguments.keys()),
                expected_output_type="dict",
            )
            for inv in invocations
        ]
        return PlannerOutput(
            required_tools=tools,
            optional_tools=[],
            alternative_tools=[],
            execution_order=tools,
            invocations=invocations,
            tool_contracts=contracts,
            rationale=(
                "Leave eligibility workflow: profile + balance + eligibility"
                + (" + policy" if include_policy else "")
                + "."
            ),
            intent_route=IntentRoute.EMPLOYEE.value,
        )

    def _plan_leave_request_draft(self, question: str) -> PlannerOutput:
        """Run eligibility tools and ask for confirmation — never auto-write."""
        leave_type, days = self._extract_leave_request_params(question)
        start, end = self._propose_dates(question, days)
        tools = [
            "get_employee_profile",
            "get_leave_balance",
            "check_leave_eligibility",
        ]
        invocations: list[ToolInvocation] = [
            ToolInvocation(tool_name="get_employee_profile", arguments={}, order=0),
            ToolInvocation(tool_name="get_leave_balance", arguments={}, order=1),
            ToolInvocation(
                tool_name="check_leave_eligibility",
                arguments={
                    "leave_type": leave_type,
                    "requested_days": days,
                },
                order=2,
            ),
        ]
        contracts = [
            ToolContract.from_golden_fields(
                tool_name=inv.tool_name,
                expected_arguments=dict(inv.arguments),
                expected_execution_order=inv.order,
                required=list(inv.arguments.keys()),
                expected_output_type="dict",
            )
            for inv in invocations
        ]
        return PlannerOutput(
            required_tools=tools,
            optional_tools=[],
            alternative_tools=[],
            execution_order=tools,
            invocations=invocations,
            tool_contracts=contracts,
            rationale=(
                "Leave request draft: validate eligibility, then require "
                "explicit human confirmation before create_leave_request."
            ),
            intent_route=IntentRoute.EMPLOYEE.value,
            pending_action={
                "type": "create_leave_request",
                "leave_type": leave_type,
                "start_date": start,
                "end_date": end,
                "reason": question.strip()[:200],
                "requested_days": days,
                "awaiting_confirmation": True,
            },
        )

    def _plan_confirmed_leave_request(
        self,
        pending: dict[str, Any],
        *,
        verified: bool,
    ) -> PlannerOutput:
        if not verified:
            return self._plan_direct(
                VERIFY_PROMPT,
                route=IntentRoute.EMPLOYEE.value,
                rationale="Confirmed leave write requires verification.",
            )
        args = {
            "leave_type": pending.get("leave_type") or "VACATION",
            "start_date": pending.get("start_date") or "",
            "end_date": pending.get("end_date") or "",
            "reason": pending.get("reason") or "Employee leave request",
            "confirmed": True,
        }
        return self._plan_single_tool(
            "create_leave_request",
            arguments=args,
            rationale="User confirmed pending leave request → write tool.",
            intent_route=IntentRoute.EMPLOYEE.value,
        )

    def _plan_manager_and_holidays(self, question: str) -> PlannerOutput:
        country, year = self._extract_country_year(question)
        holiday_args: dict[str, Any] = {}
        if country:
            holiday_args["country"] = country
        if year:
            holiday_args["year"] = year
        if not holiday_args:
            holiday_args = {"country": "US", "year": 2026}
        tools = ["get_manager_information", "get_holiday_calendar"]
        invocations = [
            ToolInvocation(
                tool_name="get_manager_information",
                arguments={},
                order=0,
            ),
            ToolInvocation(
                tool_name="get_holiday_calendar",
                arguments=holiday_args,
                order=1,
            ),
        ]
        contracts = [
            ToolContract.from_golden_fields(
                tool_name=inv.tool_name,
                expected_arguments=dict(inv.arguments),
                expected_execution_order=inv.order,
                required=list(inv.arguments.keys()),
                expected_output_type="dict",
            )
            for inv in invocations
        ]
        return PlannerOutput(
            required_tools=tools,
            optional_tools=[],
            alternative_tools=[],
            execution_order=tools,
            invocations=invocations,
            tool_contracts=contracts,
            rationale="Independent manager + holiday calendar reads.",
            intent_route=IntentRoute.EMPLOYEE.value,
        )

    def _plan_protected(
        self,
        tool_name: str,
        *,
        verified: bool,
        arguments: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        args = dict(arguments or {})
        if not verified:
            # Never execute protected tools before verification.
            return self._plan_direct(
                VERIFY_PROMPT,
                route=IntentRoute.EMPLOYEE.value,
                rationale=(
                    f"Employee data requires verification before {tool_name}."
                ),
            )
        return self._plan_single_tool(
            tool_name,
            arguments=args,
            rationale=f"Verified employee data question → {tool_name}.",
            intent_route=IntentRoute.EMPLOYEE.value,
        )

    def _plan_verify_id(self, employee_id: str) -> PlannerOutput:
        return PlannerOutput(
            required_tools=["verify_employee"],
            optional_tools=[],
            alternative_tools=[],
            execution_order=["verify_employee"],
            invocations=[
                ToolInvocation(
                    tool_name="verify_employee",
                    arguments={"employee_id": employee_id},
                    order=0,
                )
            ],
            tool_contracts=[
                ToolContract.from_golden_fields(
                    tool_name="verify_employee",
                    expected_arguments={"employee_id": employee_id},
                    expected_execution_order=0,
                    required=["employee_id"],
                    expected_output_type="dict",
                )
            ],
            rationale="Standalone employee ID → verification.",
            intent_route=IntentRoute.VERIFY_ID.value,
        )

    @staticmethod
    def _plan_direct(
        answer: str,
        *,
        route: str,
        rationale: str,
    ) -> PlannerOutput:
        return PlannerOutput(
            required_tools=[],
            optional_tools=[],
            alternative_tools=[],
            execution_order=[],
            invocations=[],
            tool_contracts=[],
            rationale=rationale,
            direct_answer=answer,
            intent_route=route,
        )

    @staticmethod
    def _plan_single_tool(
        tool_name: str,
        *,
        arguments: dict[str, Any],
        rationale: str,
        intent_route: str | None = None,
    ) -> PlannerOutput:
        return PlannerOutput(
            required_tools=[tool_name],
            optional_tools=[],
            alternative_tools=[],
            execution_order=[tool_name],
            invocations=[
                ToolInvocation(
                    tool_name=tool_name,
                    arguments=arguments,
                    order=0,
                )
            ],
            tool_contracts=[
                ToolContract.from_golden_fields(
                    tool_name=tool_name,
                    expected_arguments=arguments,
                    expected_execution_order=0,
                    required=list(arguments.keys()),
                    expected_output_type="dict",
                )
            ],
            rationale=rationale,
            intent_route=intent_route,
        )

    @staticmethod
    def _plan_search_docs_summarize(
        question: str,
        *,
        query: str,
        document: str,
        rationale: str,
        intent_route: str | None = None,
    ) -> PlannerOutput:
        """Build the canonical retrieve → summarize plan."""
        del question
        return PlannerOutput(
            required_tools=["search_docs", "summarize"],
            optional_tools=[],
            alternative_tools=["search"],
            execution_order=["search_docs", "summarize"],
            invocations=[
                ToolInvocation(
                    tool_name="search_docs",
                    arguments={"query": query},
                    order=0,
                ),
                ToolInvocation(
                    tool_name="summarize",
                    arguments={"document": document},
                    order=1,
                ),
            ],
            tool_contracts=[
                ToolContract.from_golden_fields(
                    tool_name="search_docs",
                    expected_arguments={"query": query},
                    expected_execution_order=0,
                    required=["query"],
                    maximum_calls=1,
                    minimum_calls=1,
                    maximum_latency_ms=60_000.0,
                    expected_output_type="dict",
                ),
                ToolContract.from_golden_fields(
                    tool_name="summarize",
                    expected_arguments={"document": document},
                    expected_execution_order=1,
                    required=["document"],
                    maximum_calls=1,
                    minimum_calls=1,
                    maximum_latency_ms=60_000.0,
                    expected_output_type="dict",
                ),
            ],
            rationale=rationale,
            intent_route=intent_route or IntentRoute.KNOWLEDGE.value,
        )

    @staticmethod
    def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
        return any(hint in text for hint in hints)

    @staticmethod
    def _extract_year(text: str) -> int | None:
        match = re.search(r"\b(202[4-6])\b", text)
        return int(match.group(1)) if match else None

    @classmethod
    def _looks_like_leave_balance(cls, question: str) -> bool:
        """True for personal leave-balance asks (not history / policy).

        Uses the existing hint tables plus a leave-stock + quantity composition
        so conversational follow-ups like ``So how many leaves are there?``
        resolve to ``get_leave_balance`` without a blind ``\"leave\" in query``
        shortcut.
        """
        lowered = question.lower()
        if cls._contains_any(lowered, _LEAVE_HISTORY_HINTS):
            return False
        if cls._contains_any(lowered, _UPCOMING_LEAVE_HINTS):
            return False
        if "policy" in lowered or "handbook" in lowered:
            return False
        if cls._contains_any(lowered, _LEAVE_BALANCE_HINTS):
            return True
        has_stock = cls._contains_any(lowered, _LEAVE_STOCK_TERMS)
        has_quantity = cls._contains_any(lowered, _LEAVE_BALANCE_QUANTITY_HINTS)
        return has_stock and has_quantity

    @staticmethod
    def _is_leave_policy_question(question: str) -> bool:
        lowered = question.lower()
        return any(hint in lowered for hint in _CANONICAL_LEAVE_HINTS)

    @staticmethod
    def _is_hybrid_leave_policy(question: str) -> bool:
        lowered = question.lower()
        return any(hint in lowered for hint in _HYBRID_POLICY_HINTS)

    @staticmethod
    def _looks_like_holidays(question: str) -> bool:
        lowered = question.lower()
        return any(hint in lowered for hint in _HOLIDAY_HINTS)

    @classmethod
    def _looks_like_leave_eligibility(cls, question: str) -> bool:
        """True for concrete eligibility checks, not balance inventory asks.

        ``Can I take 10 days of vacation?`` → eligibility.
        ``How many days off can I take?`` → leave balance (quantity inquiry).
        """
        lowered = question.lower()
        if cls._looks_like_leave_request(question):
            return False
        if not cls._contains_any(lowered, _LEAVE_ELIGIBILITY_HINTS):
            return False
        # Balance-style quantity questions must not become multi-tool eligibility
        # just because they contain ``can I take``.
        has_numeric_days = bool(
            re.search(r"\d+(?:\.\d+)?\s*(?:days?|day)\b", lowered)
        )
        if cls._contains_any(
            lowered,
            (
                "how many",
                "how much",
                "what's my",
                "what is my",
                "remaining",
                "left",
                "balance",
            ),
        ) and not has_numeric_days:
            return False
        return cls._contains_any(
            lowered,
            ("day", "vacation", "sick", "personal", "pto", "leave"),
        )

    @classmethod
    def _looks_like_leave_request(cls, question: str) -> bool:
        lowered = question.lower()
        if cls._contains_any(lowered, _LEAVE_REQUEST_HINTS):
            return True
        # "I want 5 days of vacation next month"
        if re.search(
            r"\b(want|like|need|request|apply|book)\b.+\b("
            r"vacation|sick|personal|pto|leave)\b",
            lowered,
        ) and re.search(r"\b\d+\b", lowered):
            return True
        return False

    @classmethod
    def _looks_like_manager_and_holidays(cls, question: str) -> bool:
        lowered = question.lower()
        has_manager = cls._contains_any(lowered, _MANAGER_HINTS)
        has_holiday = cls._contains_any(lowered, _HOLIDAY_HINTS) or "holiday" in lowered
        return has_manager and has_holiday

    @classmethod
    def _is_leave_confirmation(cls, question: str) -> bool:
        lowered = question.lower().strip()
        if cls._contains_any(lowered, _CONFIRM_HINTS):
            return True
        # Bare yes / sure when pending leave exists (checked by caller).
        return lowered in {"yes", "yep", "yup", "sure", "ok", "okay"}

    @classmethod
    def _is_leave_cancellation(cls, question: str) -> bool:
        lowered = question.lower().strip()
        if cls._contains_any(lowered, _CANCEL_HINTS):
            return True
        return lowered in {"no", "nope"}

    @classmethod
    def _extract_leave_request_params(cls, question: str) -> tuple[str, float]:
        lowered = question.lower()
        leave_type = "VACATION"
        if "sick" in lowered:
            leave_type = "SICK"
        elif "personal" in lowered:
            leave_type = "PERSONAL"
        elif "pto" in lowered or "vacation" in lowered or "annual" in lowered:
            leave_type = "VACATION"

        days = 1.0
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:days?|day)\b",
            lowered,
        )
        if match:
            days = float(match.group(1))
        else:
            match = re.search(
                r"\b(take|want|need)\s+(\d+(?:\.\d+)?)\b",
                lowered,
            )
            if match:
                days = float(match.group(2))
        return leave_type, days

    @staticmethod
    def _propose_dates(question: str, days: float) -> tuple[str, str]:
        from app.employees.service import EmployeeService

        return EmployeeService.propose_leave_window(
            requested_days=days,
            relative_hint=question,
        )

    @staticmethod
    def _extract_country_year(question: str) -> tuple[str | None, int | None]:
        lowered = question.lower()
        country = None
        if re.search(r"\b(us|usa|united states)\b", lowered):
            country = "US"
        year = None
        match = re.search(r"\b(202[4-9]|2030)\b", question)
        if match:
            year = int(match.group(1))
        return country, year

    @staticmethod
    def _looks_like_math(question: str) -> bool:
        lowered = question.lower().strip()
        if lowered.startswith("calculate") or lowered.startswith("compute"):
            return True
        stripped = question.strip()
        if not stripped:
            return False
        if re.fullmatch(r"[\d\.\s\(\)\+\-\*/×÷^%]+", stripped) and re.search(
            r"[+\-*/×÷^%]",
            stripped,
        ):
            return True
        return False

    @staticmethod
    def _looks_like_web_search(question: str) -> bool:
        lowered = question.lower()
        return any(
            token in lowered
            for token in ("search the web", "google", "look up online", "web search")
        )


class Planner:
    """Planner façade used by the LangGraph planner node.

    Args:
        strategy: Injectable planning strategy (default rule-based).
    """

    def __init__(self, strategy: PlannerStrategy | None = None) -> None:
        self._strategy = strategy or RuleBasedPlanner()

    def plan(
        self,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PlannerOutput:
        """Delegate to the configured strategy."""
        output = self._strategy.plan(question, metadata=metadata)
        logger.info(
            "Planner decided: route=%s required=%s order=%s rationale=%r",
            output.intent_route,
            output.required_tools,
            output.execution_order,
            output.rationale[:120],
        )
        return output

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """LangGraph node entrypoint."""
        question = state.get("question", "")
        metadata = dict(state.get("metadata") or {})
        try:
            planned = self.plan(question, metadata=metadata)
        except AgentPlanningError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AgentPlanningError(f"Planner failed: {exc}") from exc

        return {
            "planner_output": planned.model_dump(mode="json"),
            "selected_tools": planned.selected_tools,
            "tool_contracts": [c.model_dump(mode="json") for c in planned.tool_contracts],
            "messages": [
                {
                    "role": "planner",
                    "content": planned.rationale,
                    "selected_tools": planned.selected_tools,
                    "intent_route": planned.intent_route,
                }
            ],
            "metadata": {
                **metadata,
                "planner_prompt": planned.planner_prompt,
                "planner_response": planned.planner_response,
                "intent_route": planned.intent_route,
                "direct_answer": planned.direct_answer,
                "pending_action": planned.pending_action,
            },
        }


__all__ = [
    "PlannerStrategy",
    "RuleBasedPlanner",
    "Planner",
]
