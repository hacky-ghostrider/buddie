"""Request / response contracts for the canonical demo HTTP + CLI path."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agent.models import AgentRunResult
from app.evaluation.quality.decision import QualityDecision
from app.evaluation.report import EvaluationReport
from app.evaluation.scenarios import CANONICAL_SCENARIO_ID


class DemoRequest(BaseModel):
    """Inbound request to run the canonical interview demo.

    Attributes:
        live: When True, use real RAG / DeepEval / LangSmith (requires keys).
        offline is the default so the UI works without API keys.
        output_dir: Where evaluation / quality / benchmark artifacts are written.
    """

    model_config = ConfigDict(extra="forbid")

    live: bool = Field(
        default=False,
        description="Use live RAG / DeepEval / LangSmith when True",
    )
    output_dir: str = Field(
        default="./data/demo",
        description="Directory for demo report artifacts",
    )


class DemoResult(BaseModel):
    """Structured outcome of ``agent-tools-foundation-001`` demo run.

    Attributes:
        scenario_id: Canonical scenario identifier.
        mode: ``offline`` or ``live``.
        question: Demo question text.
        expected_tools: Tools expected by the golden example.
        agent_result: Full agent run payload.
        evaluation_report: DeepEval + tool-validation report.
        quality_decision: Quality-gate PASS / WARNING / FAIL.
        report_paths: Artifact paths keyed by kind (relative or absolute strings).
        metadata: Free-form extras.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(default=CANONICAL_SCENARIO_ID)
    mode: str = Field(description="offline | live")
    question: str
    expected_tools: list[str] = Field(default_factory=list)
    agent_result: AgentRunResult
    evaluation_report: EvaluationReport
    quality_decision: QualityDecision
    report_paths: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["DemoRequest", "DemoResult"]
