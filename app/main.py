"""FastAPI application entrypoint.

Creates the ASGI app, wires configuration and logging, and exposes a
health endpoint used by load balancers and local smoke checks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.api.v1.agent import router as agent_router
from app.api.v1.demo import router as demo_router
from app.api.v1.employees import router as employees_router
from app.api.v1.rag import router as rag_router
from app.api.v1.reports import router as reports_router
from app.config.logging import setup_logging
from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hooks (startup / shutdown).

    Args:
        _app: FastAPI application instance.

    Yields:
        Control back to the server after startup completes.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Starting %s v%s (env=%s)",
        settings.app_name,
        __version__,
        settings.app_env,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional settings override (useful in tests). When omitted,
            settings are loaded via ``get_settings()``.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = settings or get_settings()

    application = FastAPI(
        title=cfg.app_name,
        version=__version__,
        debug=cfg.app_debug,
        lifespan=lifespan,
    )

    @application.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        """Return service health status.

        Returns:
            A JSON payload with status, service name, environment, and version.
        """
        return {
            "status": "ok",
            "service": cfg.app_name,
            "environment": cfg.app_env,
            "version": __version__,
        }

    application.include_router(rag_router, prefix=cfg.api_prefix)
    application.include_router(agent_router, prefix=cfg.api_prefix)
    application.include_router(employees_router, prefix=cfg.api_prefix)
    application.include_router(demo_router, prefix=cfg.api_prefix)
    application.include_router(reports_router, prefix=cfg.api_prefix)

    return application


app = create_app()
