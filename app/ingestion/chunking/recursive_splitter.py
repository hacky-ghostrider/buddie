"""Recursive character text splitter (LangChain-compatible behaviour).

Implemented in-house so we do **not** import ``langchain_text_splitters``'s
package ``__init__``, which eagerly pulls ``sentence_transformers`` / ``torch``.
That keeps ingestion/chunking usable when torch native libs are unavailable.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.documents import Document


class RecursiveCharacterSplitter:
    """Split text by trying separators from coarsest to finest.

    Mirrors the behaviour of LangChain's ``RecursiveCharacterTextSplitter`` for
    the options we use in this platform.

    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        separators: Ordered separator preference (e.g. paragraph → line → space).
        length_function: Measures chunk length (default ``len``).
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap > chunk_size:
            raise ValueError("chunk_overlap cannot exceed chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators if separators is not None else ["\n\n", "\n", " ", ""]
        self._length_function = length_function or len

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each document's ``page_content`` and copy parent metadata.

        Args:
            documents: Source documents to split.

        Returns:
            Chunk documents with parent metadata preserved.
        """
        chunks: list[Document] = []
        for document in documents:
            for text in self.split_text(document.page_content):
                chunks.append(
                    Document(page_content=text, metadata=dict(document.metadata))
                )
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Recursively split ``text`` into chunks.

        Args:
            text: Raw input string.

        Returns:
            Ordered list of chunk strings.
        """
        return self._split_text(text, self._separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        splits = _split_on_separator(text, separator)
        good_splits: list[str] = []
        for split in splits:
            if self._length_function(split) < self._chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if new_separators:
                    final_chunks.extend(self._split_text(split, new_separators))
                else:
                    final_chunks.append(split)
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))
        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        separator_len = self._length_function(separator)
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0

        for piece in splits:
            piece_len = self._length_function(piece)
            extra = separator_len if current_doc else 0
            if total + piece_len + extra > self._chunk_size and current_doc:
                joined = separator.join(current_doc)
                if joined:
                    docs.append(joined)
                while current_doc and (
                    total > self._chunk_overlap
                    or (
                        total + piece_len + (separator_len if current_doc else 0)
                        > self._chunk_size
                        and total > 0
                    )
                ):
                    total -= self._length_function(current_doc[0]) + (
                        separator_len if len(current_doc) > 1 else 0
                    )
                    current_doc = current_doc[1:]

            current_doc.append(piece)
            total += piece_len + (separator_len if len(current_doc) > 1 else 0)

        joined = separator.join(current_doc)
        if joined:
            docs.append(joined)
        return docs


def _split_on_separator(text: str, separator: str) -> list[str]:
    """Split text on separator; empty separator splits into characters."""
    if separator:
        return [part for part in text.split(separator) if part != ""]
    return list(text)
