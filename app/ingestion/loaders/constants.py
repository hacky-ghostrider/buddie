"""Shared constants for the document loading stage.

Prefer ``MetadataKeys`` (enum) for new code. Module-level aliases remain for
backward compatibility with Sprint 2 call sites and tests.
"""

from typing import Final

from app.ingestion.metadata_keys import MetadataKeys

METADATA_SOURCE: Final[str] = MetadataKeys.SOURCE.value
METADATA_PAGE: Final[str] = MetadataKeys.PAGE.value
METADATA_FILE_NAME: Final[str] = MetadataKeys.FILE_NAME.value

__all__ = [
    "MetadataKeys",
    "METADATA_SOURCE",
    "METADATA_PAGE",
    "METADATA_FILE_NAME",
]
