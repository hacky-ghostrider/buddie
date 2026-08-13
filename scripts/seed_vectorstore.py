#!/usr/bin/env python3
"""Create the Chroma ``rag_documents`` collection and seed demo chunks.

Usage:
    uv run python scripts/seed_vectorstore.py
"""

from __future__ import annotations

import logging
import sys

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.demo.seed_corpus import ensure_demo_corpus


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)
    count = ensure_demo_corpus(settings)
    print(f"Vector store ready: collection={settings.chroma_collection_name} vectors={count}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
