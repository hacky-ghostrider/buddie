1. Where is LangGraph used?
In app/agent/graph.py:

StateGraph(AgentState) with nodes planner → router → finalize
Compiled via graph.compile()
Invoked from AgentService.run() as self._graph.invoke(initial) in app/agent/service.py.

That is the only LangGraph usage for the agent turn. Tools themselves are plain Python (ToolRegistry.execute), not LangGraph tool nodes.

2. Does RuleBasedPlanner run inside LangGraph or bypass it?
Inside LangGraph. It does not bypass the graph.

Path:

AgentService → compiled graph → planner node (Planner.__call__) → default strategy RuleBasedPlanner.plan() → writes planner_output into state → router → finalize.

Planner is a façade; RuleBasedPlanner is the injected strategy. The graph always runs.

3. What makes the planner rule-based?
RuleBasedPlanner.plan() uses deterministic heuristics, not an LLM:

Phrase/hint tables (_LEAVE_ELIGIBILITY_HINTS, _HYBRID_POLICY_HINTS, …)
Regex / substring matchers (_looks_like_leave_eligibility, _looks_like_leave_request, …)
Ordered if / first-match-wins branches that return fixed PlannerOutput (tool names, args, order, contracts)
No model call chooses tools. planner_prompt / planner_response exist on the DTO for a future LLM planner.

4. Dynamic runtime selection vs hardcoded mappings?
Hybrid, mostly hardcoded workflow templates.

Runtime: which branch fires depends on the question + metadata (verified?, pending leave?).
Not open-ended: once a branch matches, the tool list is a fixed template (e.g. eligibility → profile → balance → eligibility [→ policy if days ≥ 10]).
So: dynamic among predefined workflows, not free-form LLM tool choice.

5. Can you add a new tool without changing planner logic?
Register yes; get selected no.

ToolRegistry.register() can add a tool.
Nothing will call it unless RuleBasedPlanner gains a new heuristic / plan method that emits that tool name.
New capability ⇒ planner (and usually router answer-combining) changes.

6. What would need to change for an LLM planner later?
Designed for strategy swap:

Implement PlannerStrategy that returns PlannerOutput (optionally fill planner_prompt / planner_response).
Inject via Planner(strategy=...) / AgentService(planner=...).
Keep graph + router contracts the same.
Also needed in practice: tool schemas in the prompt, validation against the registry, verification/HITL constraints (don’t trust LLM for employee_id / write confirm), and tests for non-determinism / mocking.

Graph topology need not change for a drop-in strategy.

7. Would an LLM planner now risk the 403 tests?
Yes. Most agent tests assert exact execution_order / tool lists from RuleBasedPlanner. An LLM default would break those unless:

tests keep injecting RuleBasedPlanner, or
LLM is opt-in behind a flag with the rule-based default unchanged.
Safe path: leave RuleBasedPlanner as default; add LLM as alternate strategy only.

8. Example path: "Can I take 10 days of vacation?" (verified E-1101)
AgentService.run(...)
  → LangGraph.invoke(AgentState)
       → planner node (Planner → RuleBasedPlanner)
            classify_intent → PASS_THROUGH
            _looks_like_leave_eligibility("can i take" + days) → True
            _plan_leave_eligibility → days=10 → include policy
            PlannerOutput order:
              get_employee_profile
              get_leave_balance
              check_leave_eligibility(leave_type=VACATION, requested_days=10)
              search_company_policy(...)
       → router node
            shared_context.verified_employee_id = E-1101
            execute tools in order via ToolRegistry
            combine answers (_combine_multi_tool_answer)
       → finalize node
            EvaluationContext from ToolExecutions
  → AgentService metadata: selected_route=MULTI_TOOL, tools_invoked=[...]
If unverified, the same eligibility branch returns a direct verify prompt with no tools (still via the planner node inside LangGraph).