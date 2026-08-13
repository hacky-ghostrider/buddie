"""Chunk metadata helpers — build stable, evaluation-friendly chunk fields.

Kept separate from the chunker Strategy so enrichment rules stay testable and
reusable across future chunker implementations (token, semantic, etc.).
"""

from __future__ import annotations

from langchain_core.documents import Document

from app.ingestion.metadata_keys import MetadataKeys


def build_chunk_id(
    *,
    source: str,
    page: object | None,
    chunk_index: int,
) -> str:
    """Build a deterministic chunk identifier for citations and regression tests.

    Args:
        source: Document source path or URI.
        page: Optional page number from parent metadata.
        chunk_index: Zero-based index of this chunk in the full chunk list.

    Returns:
        Stable string id (not a random UUID) so the same inputs always match.
    """
    page_part = "none" if page is None else str(page)
    return f"{source}::p{page_part}::c{chunk_index}"


def enrich_chunk_metadata(
    *,
    page_content: str,
    parent_metadata: dict[str, object],
    chunk_index: int,
    total_chunks: int,
) -> dict[str, object]:
    """Merge parent metadata with chunk-specific fields.

    Args:
        page_content: Text of the chunk (used for measured ``chunk_size``).
        parent_metadata: Metadata copied from the source page/document.
        chunk_index: Zero-based position in the produced chunk list.
        total_chunks: Total number of chunks produced for this run.

    Returns:
        New metadata dict (does not mutate ``parent_metadata``).
    """
    metadata: dict[str, object] = dict(parent_metadata)
    source = str(metadata.get(MetadataKeys.SOURCE, "unknown"))
    page = metadata.get(MetadataKeys.PAGE)

    metadata[MetadataKeys.CHUNK_INDEX] = chunk_index
    metadata[MetadataKeys.TOTAL_CHUNKS] = total_chunks
    metadata[MetadataKeys.CHUNK_SIZE] = len(page_content)
    metadata[MetadataKeys.CHUNK_ID] = build_chunk_id(
        source=source,
        page=page,
        chunk_index=chunk_index,
    )
    return metadata


def attach_chunk_metadata(
    chunks: list[Document],
    *,
    total_chunks: int | None = None,
) -> list[Document]:
    """Attach chunk metadata to an ordered list of chunk documents.

    Args:
        chunks: Chunk documents that already preserve parent metadata.
        total_chunks: Optional override; defaults to ``len(chunks)``.

    Returns:
        New ``Document`` list with chunk metadata applied.
    """
    count = total_chunks if total_chunks is not None else len(chunks)
    enriched: list[Document] = []
    for index, chunk in enumerate(chunks):
        metadata = enrich_chunk_metadata(
            page_content=chunk.page_content,
            parent_metadata=dict(chunk.metadata),
            chunk_index=index,
            total_chunks=count,
        )
        enriched.append(Document(page_content=chunk.page_content, metadata=metadata))
    return enriched
