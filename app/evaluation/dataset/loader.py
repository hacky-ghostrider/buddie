"""Golden dataset loading — offline evaluation answer key.

WHY
---
A golden dataset is the curated exam key for AI systems: known questions,
expected answers, sources, and (for agents) expected tools. Offline
evaluation + regression depend on versioned, reviewable goldens — not
ad-hoc production logs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation.exceptions import InvalidEvaluationInputError
from app.evaluation.models import GoldenExample

logger = logging.getLogger(__name__)


class GoldenDatasetLoader:
    """Load and validate golden evaluation examples from JSON.

    Supported file shapes:
        - ``{"examples": [ ... ]}``
        - ``[ ... ]`` (bare list of examples)
    """

    def load(self, path: str | Path) -> list[GoldenExample]:
        """Load a golden dataset file.

        Args:
            path: Filesystem path to JSON dataset.

        Returns:
            Validated ``GoldenExample`` list.

        Raises:
            InvalidEvaluationInputError: Missing file or invalid schema.
        """
        dataset_path = Path(path)
        if not dataset_path.is_file():
            raise InvalidEvaluationInputError(
                f"Golden dataset not found: {dataset_path}"
            )

        try:
            raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidEvaluationInputError(
                f"Golden dataset is not valid JSON: {dataset_path}: {exc}"
            ) from exc

        records = self._extract_records(raw)
        examples: list[GoldenExample] = []
        for index, record in enumerate(records):
            try:
                examples.append(GoldenExample.model_validate(record))
            except ValidationError as exc:
                raise InvalidEvaluationInputError(
                    f"Invalid golden example at index {index} in "
                    f"{dataset_path}: {exc}"
                ) from exc

        logger.info(
            "Loaded golden dataset: path=%s count=%d",
            dataset_path,
            len(examples),
        )
        return examples

    def load_one(
        self,
        path: str | Path,
        *,
        question: str | None = None,
        example_id: str | None = None,
        index: int | None = None,
    ) -> GoldenExample:
        """Load a single example by question, id, or index.

        Args:
            path: Dataset path.
            question: Exact question match.
            example_id: Match on ``GoldenExample.id``.
            index: Zero-based index.

        Returns:
            Matching ``GoldenExample``.

        Raises:
            InvalidEvaluationInputError: When no match / ambiguous selectors.
        """
        examples = self.load(path)
        if question is not None:
            matches = [e for e in examples if e.question == question]
            if not matches:
                raise InvalidEvaluationInputError(
                    f"No golden example with question={question!r}"
                )
            return matches[0]
        if example_id is not None:
            matches = [e for e in examples if e.id == example_id]
            if not matches:
                raise InvalidEvaluationInputError(
                    f"No golden example with id={example_id!r}"
                )
            return matches[0]
        if index is not None:
            if index < 0 or index >= len(examples):
                raise InvalidEvaluationInputError(
                    f"Golden example index {index} out of range "
                    f"(size={len(examples)})"
                )
            return examples[index]
        raise InvalidEvaluationInputError(
            "Specify question, example_id, or index to load a single example"
        )

    @staticmethod
    def _extract_records(raw: Any) -> list[dict[str, Any]]:
        """Normalize file payload into a list of example dicts."""
        if isinstance(raw, list):
            return list(raw)
        if isinstance(raw, dict):
            if "examples" in raw and isinstance(raw["examples"], list):
                return list(raw["examples"])
            raise InvalidEvaluationInputError(
                "Golden dataset object must contain an 'examples' list"
            )
        raise InvalidEvaluationInputError(
            "Golden dataset must be a JSON list or object with 'examples'"
        )


__all__ = ["GoldenDatasetLoader"]
