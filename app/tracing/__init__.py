"""Tracing package — vendor-agnostic execution traces for RAG / evaluation.

LangSmith is one concrete adapter. Business code depends on ``Tracer`` /
``TracingService``, never on LangSmith client types directly.
"""

from app.tracing.base import NoOpTracer, TraceRecord, TraceSpanData, Tracer
from app.tracing.service import TracingService

__all__ = [
    "Tracer",
    "TraceRecord",
    "TraceSpanData",
    "NoOpTracer",
    "TracingService",
]
