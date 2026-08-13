"""Versioned API routers (v1)."""

from app.api.v1.agent import router as agent_router
from app.api.v1.demo import router as demo_router
from app.api.v1.employees import router as employees_router
from app.api.v1.rag import router as rag_router
from app.api.v1.reports import router as reports_router

__all__ = [
    "rag_router",
    "agent_router",
    "employees_router",
    "demo_router",
    "reports_router",
]
