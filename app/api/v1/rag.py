"""RAG query HTTP endpoints.



Endpoints stay thin: validate input, inject ``RAGService``, map domain

errors to HTTP status codes. No Retriever / LLMProvider calls here.

"""



from __future__ import annotations



import logging



from fastapi import APIRouter, Depends, HTTPException, status



from app.api.deps import get_rag_service

from app.orchestration.exceptions import (

    EmptyRAGQuestionError,

    LLMOrchestrationError,

    OrchestrationError,

    PromptOrchestrationError,

    RetrievalOrchestrationError,

    UnexpectedOrchestrationError,

)

from app.orchestration.models import RAGRequest, RAGResponse

from app.orchestration.rag_service import RAGService



logger = logging.getLogger(__name__)



router = APIRouter(prefix="/rag", tags=["rag"])





@router.post(

    "/query",

    response_model=RAGResponse,

    summary="Run an end-to-end RAG query",

)

def rag_query(

    request: RAGRequest,

    service: RAGService = Depends(get_rag_service),

) -> RAGResponse:

    """Answer a question using retrieval + grounded generation.



    Responsibilities of this endpoint:

        - Accept / validate ``RAGRequest`` (via Pydantic).

        - Delegate to ``RAGService`` (dependency injection).

        - Map domain exceptions to HTTP responses.



    Args:

        request: Inbound RAG request body.

        service: Injected orchestrator (never constructed in the handler).



    Returns:

        Structured ``RAGResponse`` with answer, evidence, and latencies.

    """

    try:

        return service.query(request)

    except EmptyRAGQuestionError as exc:

        raise HTTPException(

            status_code=status.HTTP_400_BAD_REQUEST,

            detail=str(exc),

        ) from exc

    except PromptOrchestrationError as exc:

        logger.error("RAG prompt failure via API: error=%s", exc)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    except RetrievalOrchestrationError as exc:

        logger.error("RAG retrieval failure via API: error=%s", exc)

        raise HTTPException(

            status_code=status.HTTP_502_BAD_GATEWAY,

            detail=str(exc),

        ) from exc

    except LLMOrchestrationError as exc:

        logger.error("RAG LLM failure via API: error=%s", exc)

        raise HTTPException(

            status_code=status.HTTP_502_BAD_GATEWAY,

            detail=str(exc),

        ) from exc

    except UnexpectedOrchestrationError as exc:

        logger.error("Unexpected RAG failure via API: error=%s", exc)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc

    except OrchestrationError as exc:

        logger.error("RAG orchestration failure via API: error=%s", exc)

        raise HTTPException(

            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=str(exc),

        ) from exc


