"""Pytest fixtures for document ingestion and embedding tests.

On some Windows hosts, importing real ``torch`` (pulled by sentence-transformers)
crashes with a DLL error. Tests stub those modules unless
``RAG_USE_REAL_EMBEDDINGS=1`` is set, so the suite stays runnable while still
allowing optional real-model integration later.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# IMPORTANT: stub before any app / langchain imports that may pull ST → torch.
if os.environ.get("RAG_USE_REAL_EMBEDDINGS") != "1":
    sys.modules.setdefault("torch", MagicMock(name="torch"))
    sys.modules.setdefault("sentence_transformers", MagicMock(name="sentence_transformers"))

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.config.settings import Settings


MINIMAL_PDF_WITH_TEXT = b"""%PDF-1.4
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144]
/Contents 4 0 R /Resources<< /Font<< /F1 5 0 R >> >> >>endobj
4 0 obj<< /Length 44 >>stream
BT /F1 24 Tf 100 100 Td (Hello RAG) Tj ET
endstream
endobj
5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
0000000361 00000 n 
trailer<< /Size 6 /Root 1 0 R >>
startxref
440
%%EOF
"""


@pytest.fixture
def settings() -> Settings:
    """Return ingestion settings with PDF enabled."""
    return Settings(
        app_env="test",
        supported_document_types=".pdf",
        log_level="WARNING",
    )


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    """Write a minimal one-page PDF containing extractable text."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(MINIMAL_PDF_WITH_TEXT)
    return path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Write a structurally valid PDF with zero pages."""
    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    with path.open("wb") as handle:
        writer.write(handle)
    return path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    """Write a ``.pdf`` file that is not a valid PDF."""
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"this is not a pdf file at all")
    return path


@pytest.fixture
def sample_documents() -> list[Document]:
    """Return fake LangChain documents for mocked loader tests."""
    return [
        Document(
            page_content="Page one content",
            metadata={"source": "ignored.pdf", "page": 0},
        ),
        Document(
            page_content="Page two content",
            metadata={"source": "ignored.pdf", "page": 1},
        ),
    ]
