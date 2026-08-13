"""Agent query HTTP endpoints.

Thin handlers: validate input, inject ``AgentService``, map domain errors
to HTTP status codes. No planner / tool / evaluation logic here.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.agent.exceptions import AgentError
from app.agent.models import AgentRequest, AgentRunResult
from app.agent.service import AgentService
from app.api.deps import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post(
    "/query",
    response_model=AgentRunResult,
    summary="Run one LangGraph agent turn",
)
def agent_query(
    request: AgentRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentRunResult:
    """Answer a question using the production LangGraph agent.

    Responsibilities of this endpoint:
        - Accept / validate ``AgentRequest`` (via Pydantic).
        - Delegate to ``AgentService`` (dependency injection).
        - Map domain exceptions to HTTP responses.

    Args:
        request: Inbound agent request body.
        service: Injected agent façade (never constructed in the handler).

    Returns:
        Structured ``AgentRunResult`` with answer, tools, and traces.
    """
    try:
        return service.run(
            request.question,
            metadata=request.metadata,
            expected_answer=request.expected_answer,
            expected_sources=request.expected_sources,
            validate_tools=request.validate_tools,
            correlation_id=request.correlation_id,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AgentError as exc:
        logger.error("Agent failure via API: error=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — map unexpected to 500 without stack to client
        logger.exception("Unexpected agent failure via API")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent request failed unexpectedly",
        ) from exc
