"""Canonical interview demo orchestration (thin façade over existing services)."""

from app.demo.models import DemoRequest, DemoResult
from app.demo.runner import run_canonical_demo

__all__ = ["DemoRequest", "DemoResult", "run_canonical_demo"]
