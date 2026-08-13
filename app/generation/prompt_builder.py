"""Prompt assembly for grounded generation.

Builds system + user messages from a question and retrieved documents by
loading external prompt templates. Does **not** call any LLM — pure
formatting / composition.

Why enterprise systems externalize prompts
------------------------------------------
- Prompt changes ship without redeploying application code.
- Non-engineers (prompt / domain experts) can iterate safely.
- Versioned template directories enable A/B and rollback.
- Audit / compliance can review prompt text independently of Python.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.config.settings import Settings, get_settings
from app.generation.exceptions import EmptyQuestionError, PromptTemplateError
from app.generation.models import BuiltPrompt
from app.ingestion.metadata_keys import MetadataKeys
from app.retrieval.models import RetrievedDocument

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE_NAME = "rag_system.txt"
_USER_TEMPLATE_NAME = "rag_user.txt"
_CONTEXT_PLACEHOLDER = "{{context}}"
_QUESTION_PLACEHOLDER = "{{question}}"

_EMPTY_CONTEXT_NOTICE = (
    "No retrieved documents were provided. "
    "Say that you do not have enough information to answer."
)

# Rough English heuristic when a tokenizer is unavailable (~4 chars / token).
_CHARS_PER_TOKEN_ESTIMATE = 4

# ``app/`` package root — templates default to ``app/prompts/templates``.
_APP_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class PromptBuilder:
    """Compose a grounded prompt from a question and retrieved chunks.

    Responsibilities:
        - Load system / user templates from disk (not hardcoded strings).
        - Inject retrieved context in a stable, numbered format.
        - Inject the user question.
        - Estimate prompt token count (warn-only helper for callers).
        - Never call an LLM.

    Args:
        settings: Provides ``PROMPT_TEMPLATE_DIRECTORY``.
        template_directory: Optional override for the template root.
        system_prompt: Optional override that bypasses the system template
            (useful in tests). When set, only the user template is loaded.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        template_directory: str | Path | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._template_directory = self._resolve_template_directory(
            template_directory or self._settings.prompt_template_directory
        )
        if system_prompt is not None:
            cleaned = system_prompt.strip()
            if not cleaned:
                raise ValueError("system_prompt must be a non-empty string")
            self._system_prompt_override = cleaned
        else:
            self._system_prompt_override = None

        self._system_template = (
            self._system_prompt_override
            if self._system_prompt_override is not None
            else self._load_template(_SYSTEM_TEMPLATE_NAME)
        )
        self._user_template = self._load_template(_USER_TEMPLATE_NAME)
        self._validate_user_template(self._user_template)

    def build(
        self,
        question: str,
        retrieved_documents: Sequence[RetrievedDocument] | None = None,
    ) -> BuiltPrompt:
        """Build a ``BuiltPrompt`` from question + retrieved context.

        Args:
            question: User / evaluation question text.
            retrieved_documents: Optional list of scored retrieved chunks.
                ``None`` or empty list yields an explicit empty-context notice.

        Returns:
            Formatted prompt ready for ``LLMProvider.generate``.

        Raises:
            EmptyQuestionError: Question is blank.
            PromptTemplateError: Template rendering produces blank prompts.
        """
        cleaned_question = question.strip() if question else ""
        if not cleaned_question:
            logger.error("Prompt build rejected: empty question")
            raise EmptyQuestionError("Question must be a non-empty string")

        documents = list(retrieved_documents or [])
        context_block = self._format_context(documents)
        user_prompt = self._render_user_prompt(cleaned_question, context_block)
        system_prompt = self._system_template.strip()
        if not system_prompt or not user_prompt.strip():
            raise PromptTemplateError(
                "Rendered system or user prompt is empty; check templates"
            )

        built = BuiltPrompt(
            system=system_prompt,
            user=user_prompt,
            question=cleaned_question,
            context_document_count=len(documents),
            context_char_length=len(context_block),
        )

        logger.info(
            "Prompt built: question_preview=%r context_docs=%s "
            "context_chars=%s prompt_chars=%s template_dir=%s",
            cleaned_question[:80],
            built.context_document_count,
            built.context_char_length,
            len(built.system) + len(built.user),
            self._template_directory,
        )
        return built

    def estimate_token_count(self, prompt: BuiltPrompt) -> int:
        """Estimate total prompt tokens (system + user) without calling an LLM.

        Uses a simple characters/4 heuristic so orchestration can warn when
        context approaches ``MAX_CONTEXT_TOKENS`` without adding a tokenizer
        dependency. Not a billing-grade count — provider ``usage`` remains
        authoritative after generation.

        Args:
            prompt: Built system + user prompt.

        Returns:
            Estimated token count (always ``>= 1`` for non-empty prompts).
        """
        char_count = len(prompt.system) + len(prompt.user)
        if char_count <= 0:
            return 0
        estimate = max(1, (char_count + _CHARS_PER_TOKEN_ESTIMATE - 1) // _CHARS_PER_TOKEN_ESTIMATE)
        logger.debug(
            "Prompt token estimate: chars=%s estimated_tokens=%s",
            char_count,
            estimate,
        )
        return estimate

    def _format_context(self, documents: list[RetrievedDocument]) -> str:
        """Render retrieved documents into a stable text block."""
        if not documents:
            return _EMPTY_CONTEXT_NOTICE

        parts: list[str] = []
        for index, doc in enumerate(documents, start=1):
            score_text = f"{doc.score:.4f}"
            source = (
                doc.metadata.get(MetadataKeys.FILE_NAME)
                or doc.metadata.get(MetadataKeys.SOURCE)
                or ""
            )
            header = f"[Document {index}] id={doc.id} score={score_text}"
            if source:
                header = f"{header} source={source}"
            parts.append(f"{header}\n{doc.text.strip()}")
        return "\n\n".join(parts)

    def _render_user_prompt(self, question: str, context_block: str) -> str:
        """Fill the user template placeholders with context and question."""
        rendered = self._user_template.replace(_CONTEXT_PLACEHOLDER, context_block)
        rendered = rendered.replace(_QUESTION_PLACEHOLDER, question)
        return rendered.strip()

    def _load_template(self, filename: str) -> str:
        """Load a prompt template file from the configured directory."""
        path = self._template_directory / filename
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            logger.error("Prompt template missing: path=%s", path)
            raise PromptTemplateError(
                f"Prompt template not found: {path}"
            ) from exc
        except OSError as exc:
            logger.error("Prompt template unreadable: path=%s error=%s", path, exc)
            raise PromptTemplateError(
                f"Failed to read prompt template {path}: {exc}"
            ) from exc

        cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not cleaned:
            raise PromptTemplateError(f"Prompt template is empty: {path}")
        logger.debug("Loaded prompt template: path=%s chars=%s", path, len(cleaned))
        return cleaned

    @staticmethod
    def _validate_user_template(template: str) -> None:
        """Ensure the user template exposes required placeholders."""
        missing: list[str] = []
        if _CONTEXT_PLACEHOLDER not in template:
            missing.append(_CONTEXT_PLACEHOLDER)
        if _QUESTION_PLACEHOLDER not in template:
            missing.append(_QUESTION_PLACEHOLDER)
        if missing:
            raise PromptTemplateError(
                "User prompt template missing required placeholders: "
                + ", ".join(missing)
            )

    @staticmethod
    def _resolve_template_directory(directory: str | Path) -> Path:
        """Resolve and validate the prompt template directory path."""
        path = Path(directory)
        if not path.is_absolute():
            # Try app-package-relative first (``prompts/templates``), then cwd.
            package_candidate = (_APP_PACKAGE_ROOT / path).resolve()
            cwd_candidate = path.resolve()
            if package_candidate.is_dir():
                path = package_candidate
            elif cwd_candidate.is_dir():
                path = cwd_candidate
            else:
                path = package_candidate
        if not path.is_dir():
            raise PromptTemplateError(
                f"PROMPT_TEMPLATE_DIRECTORY is not a directory: {path}"
            )
        return path

