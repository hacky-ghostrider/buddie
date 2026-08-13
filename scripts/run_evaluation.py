#!/usr/bin/env python3
"""CLI: run single-question or batch evaluation automation.

Workflow:
    Load dataset → Run RAG → LangSmith trace → DeepEval → Tool validator → Report

Examples:
    python scripts/run_evaluation.py --batch
    python scripts/run_evaluation.py --question "What is recursive character chunking?"
    python scripts/run_evaluation.py --index 0 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.evaluation.automation import EvaluationAutomationService
from app.orchestration.models import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)


class _EchoRAGRunner:
    """Deterministic offline RAG stand-in when ``--dry-run`` is set.

    Avoids live OpenAI / Chroma calls so evaluation automation can be
    exercised in CI without vendor credentials.
    """

    def query(self, request: RAGRequest) -> RAGResponse:
        from app.orchestration.models import LatencyBreakdown
        from app.retrieval.models import RetrievedDocument

        return RAGResponse(
            question=request.question.strip(),
            answer=(
                "Recursive character chunking splits text using a hierarchy of "
                "separators while preserving overlap between adjacent chunks."
            ),
            retrieved_documents=[
                RetrievedDocument(
                    id="dry-run-1",
                    text=(
                        "Recursive character chunking splits on paragraphs, "
                        "lines, spaces, then characters with overlap."
                    ),
                    metadata={"file_name": "guide.pdf"},
                    score=0.92,
                )
            ],
            retrieval_metadata={"dry_run": True},
            generation_metadata={
                "model": "dry-run",
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt": {"system": "dry-run", "user": request.question},
            },
            latency=LatencyBreakdown(
                retrieval_ms=5.0,
                prompt_build_ms=1.0,
                llm_ms=10.0,
                total_ms=16.0,
            ),
            correlation_id="dry-run",
        )


def _build_rag_runner(*, dry_run: bool):
    """Construct a real RAGService or dry-run double."""
    if dry_run:
        logger.info("Using dry-run RAG runner (no live vendor calls)")
        return _EchoRAGRunner()

    from app.api.deps import get_rag_service

    return get_rag_service()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Sprint 10 evaluation automation",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to golden_dataset.json (default from settings)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Evaluate the full golden dataset",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Evaluate a single golden question (exact match)",
    )
    parser.add_argument(
        "--example-id",
        default=None,
        help="Evaluate a single golden example by id",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Evaluate a single golden example by zero-based index",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output file stem for reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a deterministic fake RAG runner (no live APIs)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing JSON/CSV/HTML artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for evaluation automation."""
    args = parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)

    if args.dry_run:
        # Dry-run proves automation plumbing without vendor LLM-as-judge calls.
        # Tool validation is disabled: the echo RAG runner does not emit tools.
        settings = settings.model_copy(
            update={
                "enable_deepeval": False,
                "enable_langsmith": False,
                "enable_tool_validation": False,
            }
        )
        get_settings.cache_clear()

    if not args.batch and args.question is None and args.example_id is None and args.index is None:
        args.batch = True

    service = EvaluationAutomationService(
        rag_runner=_build_rag_runner(dry_run=args.dry_run),
        settings=settings,
    )
    result = service.run_from_dataset(
        args.dataset,
        question=args.question,
        example_id=args.example_id,
        index=args.index,
        write_reports=not args.no_write,
        run_name=args.run_name,
    )
    passed = sum(1 for r in result.reports if r.passed)
    logger.info(
        "Evaluation finished: total=%d passed=%d outputs=%s",
        len(result.reports),
        passed,
        {k: str(v) for k, v in result.output_paths.items()},
    )
    return 0 if passed == len(result.reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
