"""Map technical backend errors to employee-friendly copy.

Technical detail stays available in Developer / Evaluation mode; this module
never alters backend behavior.
"""

from __future__ import annotations

from frontend.api_client import ApiClientError

_FRIENDLY_DEFAULT = (
    "Sorry, I couldn't complete that request right now. Please try again."
)

_KNOWLEDGE_BASE = (
    "Sorry, I couldn't access the company knowledge base right now. "
    "Please try again."
)

_UNAVAILABLE = (
    "Sorry, Buddie is temporarily unavailable. Please try again in a moment."
)

_TIMEOUT = "That's taking longer than expected. Please try again."

_VERIFY_FAILED = (
    "That employee ID couldn't be verified.\n\n"
    "Employee IDs should follow the format E-1101.\n"
    "Please recheck and try again."
)


def friendly_error(exc: Exception) -> str:
    """Return a non-technical message safe to show in the main chat."""
    if isinstance(exc, ApiClientError):
        text = (exc.message or "").lower()
        if exc.status_code in {400, 404} and (
            "employee" in text or "verif" in text or "not found" in text
        ):
            return _VERIFY_FAILED
        if any(
            token in text
            for token in (
                "collection",
                "does not exist",
                "retrieval",
                "vector",
                "chroma",
                "knowledge",
            )
        ):
            return _KNOWLEDGE_BASE
        if "unavailable" in text or "connection" in text or "connect" in text:
            return _UNAVAILABLE
        if "timed out" in text or "timeout" in text:
            return _TIMEOUT
        # Prefer not to surface FastAPI / uvicorn / stack phrasing.
        if any(
            token in text
            for token in ("traceback", "exception", "uvicorn", "fastapi", "status 5")
        ):
            return _FRIENDLY_DEFAULT
        # Keep short validation hints; hide long internal details.
        if len(exc.message) <= 120 and "rag_" not in text and "chrom" not in text:
            return exc.message
    return _FRIENDLY_DEFAULT


def technical_detail(exc: Exception) -> str:
    """Return raw detail for Developer / Evaluation mode."""
    if isinstance(exc, ApiClientError):
        status = f" (HTTP {exc.status_code})" if exc.status_code else ""
        return f"{exc.message}{status}"
    return str(exc)
