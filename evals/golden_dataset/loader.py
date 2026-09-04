"""Load the Sprint 17 Buddie golden dataset without touching Sprint 9 loaders."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evals.golden_dataset.models import BuddieGoldenCase, BuddieGoldenDataset, BuddieTestTier

BUDDIE_GOLDEN_CASES_PATH = Path(__file__).resolve().parent / "buddie_golden_cases.json"


def load_buddie_golden_dataset(
    path: str | Path | None = None,
) -> BuddieGoldenDataset:
    """Load and validate ``buddie_golden_cases.json``.

    Args:
        path: Optional override path (defaults to the Sprint 17 baseline file).

    Returns:
        Validated ``BuddieGoldenDataset``.

    Raises:
        FileNotFoundError: When the dataset file is missing.
        ValueError: When JSON shape / case schema is invalid.
    """
    dataset_path = Path(path) if path is not None else BUDDIE_GOLDEN_CASES_PATH
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Buddie golden dataset not found: {dataset_path}")

    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Buddie golden dataset is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Buddie golden dataset root must be a JSON object")
    if "cases" not in raw or not isinstance(raw["cases"], list):
        raise ValueError("Buddie golden dataset must contain a 'cases' list")

    try:
        return BuddieGoldenDataset.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid Buddie golden dataset: {exc}") from exc


def filter_cases_by_tier(
    cases: list[BuddieGoldenCase],
    tier: BuddieTestTier,
) -> list[BuddieGoldenCase]:
    """Return golden cases tagged for the given CI tier."""
    return [case for case in cases if tier in case.test_tier]


def case_ids_for_tier(
    dataset: BuddieGoldenDataset,
    tier: BuddieTestTier,
) -> list[str]:
    """Stable ordered case ids for a tier (preserves dataset order)."""
    return [case.id for case in filter_cases_by_tier(dataset.cases, tier)]


__all__ = [
    "BUDDIE_GOLDEN_CASES_PATH",
    "BuddieGoldenCase",
    "BuddieGoldenDataset",
    "BuddieTestTier",
    "case_ids_for_tier",
    "filter_cases_by_tier",
    "load_buddie_golden_dataset",
]
