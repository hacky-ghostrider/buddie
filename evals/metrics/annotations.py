"""Annotation / ground-truth coverage summary for Buddie golden cases."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evals.golden_dataset.loader import load_buddie_golden_dataset
from evals.golden_dataset.models import BuddieGoldenDataset

EXPECTED_CATEGORY_COUNTS: dict[str, int] = {
    "leave_hr": 8,
    "holidays": 4,
    "benefits_policies": 3,
    "rag_knowledge": 4,
    "multi_tool": 5,
    "negative_unknown": 4,
    "adversarial_security": 8,
}


class AnnotationCoverageReport(BaseModel):
    """Interview-friendly annotation summary over the 28 goldens."""

    model_config = ConfigDict(extra="forbid")

    total_cases: int
    by_category: dict[str, int] = Field(default_factory=dict)
    cases_with_expected_tool: list[str] = Field(default_factory=list)
    cases_with_expected_tools: list[str] = Field(default_factory=list)
    cases_with_expected_context: list[str] = Field(default_factory=list)
    cases_with_expected_answer: list[str] = Field(default_factory=list)
    cases_with_hitl_expectation: list[str] = Field(default_factory=list)
    cases_with_verification_expectation: list[str] = Field(default_factory=list)
    cases_with_negative_unknown_behavior: list[str] = Field(default_factory=list)
    cases_with_evaluation_notes: list[str] = Field(default_factory=list)
    cases_with_adversarial_security: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_annotation_report(
    dataset: BuddieGoldenDataset | None = None,
) -> AnnotationCoverageReport:
    """Summarize which annotation fields each golden case provides."""
    data = dataset or load_buddie_golden_dataset()
    by_category = dict(Counter(c.category for c in data.cases))

    with_tool: list[str] = []
    with_tools: list[str] = []
    with_context: list[str] = []
    with_answer: list[str] = []
    with_hitl: list[str] = []
    with_verification: list[str] = []
    with_negative: list[str] = []
    with_adversarial: list[str] = []
    with_notes: list[str] = []

    for case in data.cases:
        with_answer.append(case.id)
        if case.expected_tool:
            with_tool.append(case.id)
        if case.expected_tools:
            with_tools.append(case.id)
        if case.expected_context:
            with_context.append(case.id)
        if case.expected_behavior == "require_hitl_confirmation":
            with_hitl.append(case.id)
        if case.expected_behavior == "require_verification":
            with_verification.append(case.id)
        if (
            case.category == "negative_unknown"
            or case.expected_behavior
            in {"refuse_or_insufficient", "require_verification"}
        ):
            with_negative.append(case.id)
        if case.category == "adversarial_security":
            with_adversarial.append(case.id)
        if case.evaluation_notes:
            with_notes.append(case.id)

    return AnnotationCoverageReport(
        total_cases=len(data.cases),
        by_category=by_category,
        cases_with_expected_tool=with_tool,
        cases_with_expected_tools=with_tools,
        cases_with_expected_context=with_context,
        cases_with_expected_answer=with_answer,
        cases_with_hitl_expectation=with_hitl,
        cases_with_verification_expectation=with_verification,
        cases_with_negative_unknown_behavior=with_negative,
        cases_with_adversarial_security=with_adversarial,
        cases_with_evaluation_notes=with_notes,
        counts={
            "expected_tool": len(with_tool),
            "expected_tools": len(with_tools),
            "expected_context": len(with_context),
            "expected_answer": len(with_answer),
            "hitl_expectation": len(with_hitl),
            "verification_expectation": len(with_verification),
            "negative_unknown_behavior": len(with_negative),
            "adversarial_security": len(with_adversarial),
            "evaluation_notes": len(with_notes),
        },
    )


def format_annotation_console(report: AnnotationCoverageReport) -> str:
    """Concise console annotation summary."""
    lines = [
        "Buddie Annotation Summary",
        f"  Total cases: {report.total_cases}",
        "  By category:",
    ]
    for name in (
        "leave_hr",
        "holidays",
        "benefits_policies",
        "rag_knowledge",
        "multi_tool",
        "negative_unknown",
        "adversarial_security",
    ):
        lines.append(f"    {name}: {report.by_category.get(name, 0)}")
    lines.extend(
        [
            "  Annotation coverage:",
            f"    expected tool: {report.counts.get('expected_tool', 0)}",
            f"    expected tools: {report.counts.get('expected_tools', 0)}",
            f"    expected context: {report.counts.get('expected_context', 0)}",
            f"    expected answer: {report.counts.get('expected_answer', 0)}",
            f"    HITL expectation: {report.counts.get('hitl_expectation', 0)}",
            f"    verification expectation: "
            f"{report.counts.get('verification_expectation', 0)}",
            f"    negative/unknown behavior: "
            f"{report.counts.get('negative_unknown_behavior', 0)}",
            f"    adversarial/security: "
            f"{report.counts.get('adversarial_security', 0)}",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "EXPECTED_CATEGORY_COUNTS",
    "AnnotationCoverageReport",
    "build_annotation_report",
    "format_annotation_console",
]
