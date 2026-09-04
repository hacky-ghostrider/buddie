"""Pydantic models for Buddie golden cases (Sprint 17 schema).

Separate from ``app.evaluation.models.GoldenExample`` (Sprint 9–12).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BuddieCategory = Literal[
    "leave_hr",
    "holidays",
    "benefits_policies",
    "rag_knowledge",
    "multi_tool",
    "negative_unknown",
    "adversarial_security",
]

BuddieExpectedBehavior = Literal[
    "answer_from_tool",
    "answer_from_rag",
    "combine_tools",
    "refuse_or_insufficient",
    "require_verification",
    "require_hitl_confirmation",
]

BuddieTestTier = Literal["smoke", "sanity", "regression"]


class BuddieGoldenCase(BaseModel):
    """One Buddie golden evaluation case from ``buddie_golden_cases.json``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable unique case id")
    category: BuddieCategory
    user_query: str
    expected_answer: str
    expected_behavior: BuddieExpectedBehavior
    expected_context: list[str] = Field(default_factory=list)
    expected_tool: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    evaluation_notes: str | None = None
    test_tier: list[BuddieTestTier] = Field(
        default_factory=lambda: ["regression"],
        description="CI tiers this case belongs to (smoke, sanity, regression)",
    )

    @field_validator("test_tier")
    @classmethod
    def _test_tier_non_empty_and_valid(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("test_tier must contain at least one tier")
        allowed = {"smoke", "sanity", "regression"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown test_tier values: {sorted(unknown)}")
        if "regression" not in value:
            raise ValueError("every case must include regression in test_tier")
        return value

    @field_validator("id", "user_query", "expected_answer")
    @classmethod
    def _required_text_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("required text fields must be non-empty")
        return cleaned


class BuddieGoldenDataset(BaseModel):
    """File-level Buddie golden dataset document."""

    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    description: str | None = None
    default_session: dict[str, Any] = Field(default_factory=dict)
    cases: list[BuddieGoldenCase]

    @field_validator("name", "version")
    @classmethod
    def _identity_non_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("dataset identity fields must be non-empty")
        return cleaned


__all__ = [
    "BuddieCategory",
    "BuddieExpectedBehavior",
    "BuddieGoldenCase",
    "BuddieGoldenDataset",
    "BuddieTestTier",
]
