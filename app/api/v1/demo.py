"""Canonical demo HTTP endpoint.

Delegates to ``run_canonical_demo`` — no evaluation logic in the handler.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.demo.models import DemoRequest, DemoResult
from app.demo.runner import run_canonical_demo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/demo", tags=["demo"])


@router.post(
    "/run",
    response_model=DemoResult,
    summary="Run agent-tools-foundation-001 canonical demo",
)
def run_demo(request: DemoRequest = DemoRequest()) -> DemoResult:
    """Execute the interview demonstration pipeline end-to-end.

    Offline by default (deterministic DeepEval + NoOpTracer). Set
    ``live=true`` only when OpenAI / LangSmith keys are configured.

    Args:
        request: Demo options (live flag, output directory).

    Returns:
        Structured ``DemoResult`` with agent, evaluation, and quality data.
    """
    try:
        return run_canonical_demo(
            live=request.live,
            output_dir=request.output_dir,
        )
    except FileNotFoundError as exc:
        logger.error("Demo dataset missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Demo failed via API")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Demo pipeline failed unexpectedly",
        ) from exc
