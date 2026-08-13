"""Report listing / viewing HTTP endpoints.

Exposes existing report files under configured directories. Does not
create a second reporting system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import get_api_settings
from app.config.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

ReportKind = Literal[
    "evaluation",
    "quality",
    "benchmark",
    "demo_evaluation",
    "demo_quality",
    "demo_benchmark",
]

_KIND_TO_SETTING: dict[str, str] = {
    "evaluation": "report_directory",
    "quality": "quality_report_directory",
    "benchmark": "benchmark_directory",
    "demo_evaluation": "demo_reports",
    "demo_quality": "demo_quality",
    "demo_benchmark": "demo_benchmarks",
}

_ALLOWED_SUFFIXES = {".json", ".csv", ".html", ".md"}


class ReportFileInfo(BaseModel):
    """Metadata for one report artifact on disk."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    path: str
    suffix: str
    size_bytes: int


class ReportListResponse(BaseModel):
    """List of available report files."""

    model_config = ConfigDict(extra="forbid")

    files: list[ReportFileInfo] = Field(default_factory=list)


class ReportContentResponse(BaseModel):
    """File content payload for viewing in the UI."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    name: str
    path: str
    content_type: str
    content: str | dict[str, Any]


def _resolve_kind_dir(settings: Settings, kind: str) -> Path:
    """Map a report kind to a filesystem directory."""
    if kind == "evaluation":
        return Path(settings.report_directory)
    if kind == "quality":
        return Path(settings.quality_report_directory)
    if kind == "benchmark":
        return Path(settings.benchmark_directory)
    if kind == "demo_evaluation":
        return Path("./data/demo/reports")
    if kind == "demo_quality":
        return Path("./data/demo/quality")
    if kind == "demo_benchmark":
        return Path("./data/demo/benchmarks")
    raise ValueError(f"Unknown report kind: {kind}")


def _safe_resolve(base: Path, name: str) -> Path:
    """Resolve ``name`` under ``base`` without path traversal."""
    candidate = (base / name).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report path",
        )
    return candidate


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List latest evaluation / quality / benchmark reports",
)
def list_reports(
    kind: ReportKind | None = Query(
        default=None,
        description="Optional filter: evaluation | quality | benchmark | demo_*",
    ),
    settings: Settings = Depends(get_api_settings),
) -> ReportListResponse:
    """List report files from configured artifact directories."""
    kinds: list[str]
    if kind is None:
        kinds = list(_KIND_TO_SETTING.keys())
    else:
        kinds = [kind]

    files: list[ReportFileInfo] = []
    for k in kinds:
        directory = _resolve_kind_dir(settings, k)
        if not directory.exists():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_SUFFIXES:
                continue
            files.append(
                ReportFileInfo(
                    kind=k,
                    name=path.name,
                    path=str(path),
                    suffix=path.suffix.lower(),
                    size_bytes=path.stat().st_size,
                )
            )
    return ReportListResponse(files=files)


@router.get(
    "/content",
    response_model=ReportContentResponse,
    summary="Read one report file for UI display",
)
def read_report(
    kind: ReportKind = Query(description="Report kind"),
    name: str = Query(description="File name within the kind directory"),
    settings: Settings = Depends(get_api_settings),
) -> ReportContentResponse:
    """Return report content (JSON parsed, otherwise text)."""
    if not name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report name",
        )
    directory = _resolve_kind_dir(settings, kind)
    path = _safe_resolve(directory, name)
    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report not found: {name}",
        )
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported report type",
        )

    text = path.read_text(encoding="utf-8")
    content: str | dict[str, Any] = text
    content_type = "text/plain"
    if path.suffix.lower() == ".json":
        import json

        try:
            content = json.loads(text)
            content_type = "application/json"
        except json.JSONDecodeError:
            content_type = "text/plain"
    elif path.suffix.lower() == ".html":
        content_type = "text/html"
    elif path.suffix.lower() == ".csv":
        content_type = "text/csv"

    return ReportContentResponse(
        kind=kind,
        name=name,
        path=str(path),
        content_type=content_type,
        content=content,
    )
