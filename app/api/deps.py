"""FastAPI dependency providers for the HTTP layer.

Keeps route handlers thin: endpoints receive injected services and never
construct OpenAI / Chroma clients themselves.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from app.agent.service import AgentService
from app.config.settings import Settings, get_settings
from app.demo.seed_corpus import ensure_demo_corpus
from app.embeddings.factory import build_embedding_model
from app.employees.service import EmployeeService
from app.employees.store import EmployeeStore
from app.generation.offline_provider import build_llm_provider
from app.generation.prompt_builder import PromptBuilder
from app.orchestration.rag_service import RAGService
from app.retrieval.vector_retriever import VectorRetriever
from app.tracing.service import TracingService
from app.vectorstore.factory import build_vector_store

logger = logging.getLogger(__name__)


@lru_cache
def get_rag_service() -> RAGService:
    """Build a cached ``RAGService`` wired to production collaborators.

    Composition root for the RAG query path. Unit tests should override this
    dependency with a mocked ``RAGService`` so CI never loads embedding weights
    or calls OpenAI / Chroma.

    Returns:
        Fully wired orchestrator instance.
    """
    settings = get_settings()
    logger.info(
        "Wiring RAGService: embedding_model=%s collection=%s "
        "prompt_templates=%s openai_model=%s",
        settings.embedding_model,
        settings.chroma_collection_name,
        settings.prompt_template_directory,
        settings.openai_model,
    )
    embedding_model = build_embedding_model(settings)
    vector_store = build_vector_store(settings)
    # Live UI /api/v1/rag/query needs a real collection; offline make demo mocks RAG.
    ensure_demo_corpus(
        settings,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    retriever = VectorRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        settings=settings,
    )
    prompt_builder = PromptBuilder(settings=settings)
    llm_provider = build_llm_provider(settings)
    return RAGService(
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        settings=settings,
    )


@lru_cache
def get_employee_service() -> EmployeeService:
    """Build a cached ``EmployeeService`` over the JSON employee store.

    Ensures the deterministic ~30-employee dataset exists on first use.
    """
    settings = get_settings()
    store = EmployeeStore(settings.employee_data_path)
    store.ensure_seeded()
    logger.info(
        "Wiring EmployeeService: path=%s",
        settings.employee_data_path,
    )
    return EmployeeService(store=store)


@lru_cache
def get_agent_service() -> AgentService:
    """Build a cached ``AgentService`` on top of the shared ``RAGService``.

    Composition root for the agent query path. Unit tests should override this
    dependency with a mocked ``AgentService`` so CI never loads embeddings or
    calls OpenAI / Chroma / LangSmith.

    Returns:
        Fully wired agent façade.
    """
    settings = get_settings()
    logger.info("Wiring AgentService (reuses RAGService + TracingService)")
    return AgentService(
        rag_service=get_rag_service(),
        employee_service=get_employee_service(),
        tracing_service=TracingService(settings=settings),
        settings=settings,
    )


def get_api_settings() -> Settings:
    """Provide application settings to route handlers."""
    return get_settings()

