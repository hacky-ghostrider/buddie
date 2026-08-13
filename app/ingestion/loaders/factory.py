"""Document loader factory and extension registry."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import TypeAlias

from app.config.settings import Settings, get_settings
from app.ingestion.loaders.base import DocumentLoader
from app.ingestion.loaders.exceptions import UnsupportedDocumentTypeError
from app.ingestion.loaders.pdf_loader import PDFDocumentLoader

logger = logging.getLogger(__name__)

LoaderBuilder: TypeAlias = Callable[[Path, Settings], DocumentLoader]


def _build_pdf_loader(file_path: Path, settings: Settings) -> DocumentLoader:
    """Build a ``PDFDocumentLoader`` for the given path."""
    return PDFDocumentLoader(file_path=file_path, settings=settings)


DEFAULT_LOADER_REGISTRY: dict[str, LoaderBuilder] = {
    ".pdf": _build_pdf_loader,
}


class DocumentLoaderFactory:
    """Create ``DocumentLoader`` instances from a path using a registry.

    Args:
        settings: Controls which extensions are allowed by configuration.
        registry: Optional map of ``extension -> builder``. Defaults to
            ``DEFAULT_LOADER_REGISTRY``. Pass a custom map in tests.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        registry: Mapping[str, LoaderBuilder] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._registry: MutableMapping[str, LoaderBuilder] = dict(
            registry if registry is not None else DEFAULT_LOADER_REGISTRY
        )

    def create(self, file_path: str | Path) -> DocumentLoader:
        """Resolve and construct a loader for ``file_path``.

        Args:
            file_path: Local document path whose suffix selects the loader.

        Returns:
            A concrete ``DocumentLoader`` ready for ``load()``.

        Raises:
            UnsupportedDocumentTypeError: Extension not allowed by settings
                or not present in the registry.
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        allowed = self._settings.get_supported_extensions()

        if extension not in allowed:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{extension}' for {path}. "
                f"Allowed: {sorted(allowed)}"
            )

        builder = self._registry.get(extension)
        if builder is None:
            raise UnsupportedDocumentTypeError(
                f"No loader registered for extension '{extension}' (path={path}). "
                f"Registered: {sorted(self._registry)}"
            )

        loader = builder(path, self._settings)
        logger.info(
            "Factory created loader=%s for path=%s",
            type(loader).__name__,
            path,
        )
        return loader

    def register(self, extension: str, builder: LoaderBuilder) -> None:
        """Register or replace a loader builder for an extension.

        Args:
            extension: File extension (with or without leading dot).
            builder: Callable that builds a ``DocumentLoader``.
        """
        normalized = extension.strip().lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        self._registry[normalized] = builder
        logger.info("Registered loader for extension=%s", normalized)

    @property
    def registered_extensions(self) -> frozenset[str]:
        """Return the set of extensions currently in the registry."""
        return frozenset(self._registry)
