# Golden Dataset Catalog

## Golden Dataset v1 — Sprint 17

These **28** cases in `buddie_golden_cases.json` are the **frozen baseline** for upcoming DeepEval evaluation work (functional correctness, RAG/groundedness, response quality).

Do not change case IDs, queries, expected answers, expected tools, expected context, or expected behavior without an explicit Sprint decision to revise the baseline.

Source of truth: [`buddie_golden_cases.json`](buddie_golden_cases.json)

Default session persona: verified employee **E-1101** (Avery Nguyen), unless a case notes otherwise.

## Summary

Total cases: 28

Categories:
- leave_hr: 8
- holidays: 4
- benefits_policies: 3
- rag_knowledge: 4
- multi_tool: 5
- negative_unknown: 4

## Case index (1–28)

| # | Case ID | Category | User Query | Expected Tool(s) | Expected Behavior |
|---|---------|----------|------------|-------------------|-------------------|
| 1 | `leave-balance-vacation-001` | `leave_hr` | How many vacation days do I have left? | get_leave_balance | `answer_from_tool` |
| 2 | `leave-balance-sick-002` | `leave_hr` | What is my sick leave balance? | get_leave_balance | `answer_from_tool` |
| 3 | `leave-balance-personal-003` | `leave_hr` | How many personal leave days do I have? | get_leave_balance | `answer_from_tool` |
| 4 | `leave-history-2025-004` | `leave_hr` | Show my leave history for 2025 | get_leave_history | `answer_from_tool` |
| 5 | `leave-upcoming-005` | `leave_hr` | What is my upcoming leave? | get_upcoming_leave | `answer_from_tool` |
| 6 | `leave-eligibility-ok-006` | `leave_hr` | Am I eligible to take 3 vacation days? | get_employee_profile, get_leave_balance, check_leave_eligibility | `combine_tools` |
| 7 | `leave-eligibility-insufficient-007` | `leave_hr` | Can I take 30 vacation days? | get_employee_profile, get_leave_balance, check_leave_eligibility, search_company_policy | `combine_tools` |
| 8 | `leave-profile-008` | `leave_hr` | Show my profile | get_employee_profile | `answer_from_tool` |
| 9 | `holiday-next-009` | `holidays` | What is the next company holiday? | get_upcoming_holidays | `answer_from_tool` |
| 10 | `holiday-upcoming-list-010` | `holidays` | What holidays are coming up? | get_upcoming_holidays | `answer_from_tool` |
| 11 | `holiday-calendar-us-2026-011` | `holidays` | Show me the US holiday calendar for 2026 | get_holiday_calendar | `answer_from_tool` |
| 12 | `holiday-labor-day-012` | `holidays` | When is Labor Day 2026 on the holiday calendar? | get_holiday_calendar | `answer_from_tool` |
| 13 | `policy-leave-handbook-013` | `benefits_policies` | What is the company leave policy for vacation? | search_docs, summarize | `answer_from_rag` |
| 14 | `policy-parental-handbook-014` | `benefits_policies` | I need parental leave information from the handbook | search_docs, summarize | `answer_from_rag` |
| 15 | `policy-benefits-limited-015` | `benefits_policies` | What employee benefits does the company offer? | search_docs, summarize | `answer_from_rag` |
| 16 | `rag-summarize-leave-policy-016` | `rag_knowledge` | Summarize the leave policy from the employee handbook. | search_docs, summarize | `answer_from_rag` |
| 17 | `rag-carry-forward-cap-017` | `rag_knowledge` | What does the employee handbook say is the vacation carry-forward maximum? | search_docs, summarize | `answer_from_rag` |
| 18 | `rag-notice-requirement-018` | `rag_knowledge` | What does the employee handbook say about vacation approval and notice? | search_docs, summarize | `answer_from_rag` |
| 19 | `rag-buddie-identity-019` | `rag_knowledge` | What does the handbook say Buddie is? | search_docs, summarize | `answer_from_rag` |
| 20 | `multi-carry-forward-020` | `multi_tool` | Can I carry forward my remaining vacation days? | get_leave_balance, search_company_policy | `combine_tools` |
| 21 | `multi-manager-holidays-021` | `multi_tool` | Who is my manager and what holidays are coming up? | get_manager_information, get_holiday_calendar | `combine_tools` |
| 22 | `multi-eligibility-with-policy-022` | `multi_tool` | Can I take 10 vacation days according to policy? | get_employee_profile, get_leave_balance, check_leave_eligibility, search_company_policy | `combine_tools` |
| 23 | `multi-leave-request-hitl-023` | `multi_tool` | I want to request 3 vacation days starting 2026-10-01 | get_employee_profile, get_leave_balance, check_leave_eligibility | `require_hitl_confirmation` |
| 24 | `negative-external-ceo-024` | `negative_unknown` | What is the CEO salary of Apple? | —(none) | `refuse_or_insufficient` |
| 25 | `negative-unverified-balance-025` | `negative_unknown` | How many vacation days do I have left? | —(none) | `require_verification` |
| 26 | `negative-unsupported-stipend-026` | `negative_unknown` | What is our remote work stipend amount for pets? | search_docs, summarize | `refuse_or_insufficient` |
| 27 | `negative-no-invented-holiday-027` | `negative_unknown` | Is Diwali an official company holiday in our US calendar for 2026? | get_holiday_calendar | `refuse_or_insufficient` |
| 28 | `multi-sick-carry-forward-028` | `multi_tool` | Does sick leave carry forward? | get_leave_balance, search_company_policy | `combine_tools` |

## Notes

- Row numbers are catalog indexes only; stable identifiers are the **Case ID** values.
- Empty `expected_tools` means no tool call is expected (shown as `—(none)`).
- Full expected answers and context live in the JSON; this README is an index, not a second source of truth.
