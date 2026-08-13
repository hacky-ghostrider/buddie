"""Generation layer base exports.

Re-exports ``LLMProvider`` so imports match other stages
(``from app.generation.base import LLMProvider``), while the ABC itself
lives in ``llm_provider.py`` for a clear file name.
"""

from app.generation.llm_provider import LLMProvider

__all__ = ["LLMProvider"]
