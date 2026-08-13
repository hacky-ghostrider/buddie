"""Canonical agent-tools foundation scenario constants (Sprint 10.2).

``agent-tools-foundation-001`` is the official demonstration scenario.
It stays unchanged across Sprint 11 so interviews can show evolution:

* Sprint 10.2 — expected tools set, actual tools empty (no agent yet)
* Sprint 11 — same expectations, actual tools populated by LangGraph
"""

from __future__ import annotations

from pathlib import Path

CANONICAL_SCENARIO_ID = "agent-tools-foundation-001"
CANONICAL_DATASET_PATH = Path("datasets") / "agent-tools-foundation-001.json"

# Sprint 10.2 reality: infrastructure only — no agent runtime.
SPRINT_10_2_ACTUAL_TOOLS: list[str] = []

# Sprint 11 behaviour: LangGraph agent populates these executions.
SPRINT_11_EXPECTED_ACTUAL_TOOLS: list[str] = ["search_docs", "summarize"]


__all__ = [
    "CANONICAL_SCENARIO_ID",
    "CANONICAL_DATASET_PATH",
    "SPRINT_10_2_ACTUAL_TOOLS",
    "SPRINT_11_EXPECTED_ACTUAL_TOOLS",
]
