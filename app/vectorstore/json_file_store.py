"""JSON file vector store — Windows-safe brute-force cosine backend.

Chroma's native ``collection.add`` can access-violate on some Windows hosts
(Rust bindings without SegmentAPI/hnswlib). This Strategy keeps the same
``VectorStore`` contract with a simple JSON persistence file so local UI
demos still work.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from app.config.settings import Settings, get_settings
from app.embeddings.models import EmbeddedDocument
from app.ingestion.metadata_keys import MetadataKeys
from app.retrieval.models import RetrievedDocument
from app.vectorstore.base import VectorStore
from app.vectorstore.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DuplicateDocumentIdError,
    EmptyDocumentListError,
    InvalidEmbeddingDimensionError,
    InvalidVectorStoreConfigError,
    MissingDocumentIdError,
    VectorStorePersistenceError,
)
from app.vectorstore.models import AddDocumentsResult

logger = logging.getLogger(__name__)

_ID_METADATA_KEYS: tuple[str, ...] = (
    MetadataKeys.CHUNK_ID,
    "id",
    "document_id",
)


class JsonFileVectorStore(VectorStore):
    """Persist vectors as JSON and search with brute-force cosine similarity."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._expected_dimension: int | None = None
        self._validate_config()
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

    @property
    def collection_name(self) -> str:
        return self._settings.chroma_collection_name

    @property
    def persist_directory(self) -> str:
        return self._settings.chroma_persist_directory

    @property
    def _store_path(self) -> Path:
        return Path(self.persist_directory) / f"{self.collection_name}.json"

    def create_collection(self) -> None:
        if self.collection_exists():
            raise CollectionAlreadyExistsError(
                f"Collection '{self.collection_name}' already exists"
            )
        self._write_payload({"name": self.collection_name, "documents": []})
        logger.info("JSON collection created: name=%s path=%s", self.collection_name, self._store_path)

    def add_documents(self, documents: list[EmbeddedDocument]) -> AddDocumentsResult:
        if not documents:
            raise EmptyDocumentListError("Cannot add an empty document list")
        payload = self._read_payload()
        existing_ids = {str(item["id"]) for item in payload["documents"]}
        records: list[dict[str, Any]] = []
        dimensions: set[int] = set()

        for index, document in enumerate(documents):
            doc_id = _resolve_document_id(document, index=index)
            if doc_id in existing_ids or any(r["id"] == doc_id for r in records):
                raise DuplicateDocumentIdError(f"Duplicate document id: '{doc_id}'")
            embedding = list(document.embedding)
            if not embedding:
                raise InvalidEmbeddingDimensionError("Embedding must be non-empty")
            dimensions.add(len(embedding))
            records.append(
                {
                    "id": doc_id,
                    "text": document.text,
                    "embedding": embedding,
                    "metadata": dict(document.metadata or {}),
                }
            )

        if len(dimensions) != 1:
            raise InvalidEmbeddingDimensionError(
                f"Inconsistent embedding dimensions in batch: {sorted(dimensions)}"
            )
        dimension = next(iter(dimensions))
        if self._expected_dimension is None and payload["documents"]:
            self._expected_dimension = len(payload["documents"][0]["embedding"])
        if self._expected_dimension is None:
            self._expected_dimension = dimension
        elif dimension != self._expected_dimension:
            raise InvalidEmbeddingDimensionError(
                f"Embedding dimension {dimension} does not match "
                f"collection dimension {self._expected_dimension}"
            )

        payload["documents"].extend(records)
        self._write_payload(payload)
        logger.info(
            "JSON documents add finished: added=%s total=%s",
            len(records),
            len(payload["documents"]),
        )
        return AddDocumentsResult(
            ids=[record["id"] for record in records],
            added_count=len(records),
            collection_name=self.collection_name,
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        score_threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidVectorStoreConfigError(
                f"top_k must be a positive integer, got {top_k!r}"
            )
        threshold = float(score_threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise InvalidVectorStoreConfigError(
                f"score_threshold must be between 0 and 1 inclusive, got {threshold}"
            )
        if not query_embedding:
            raise InvalidEmbeddingDimensionError("query_embedding must be non-empty")

        payload = self._read_payload()
        results: list[RetrievedDocument] = []
        for item in payload["documents"]:
            metadata = dict(item.get("metadata") or {})
            if metadata_filter and any(
                metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            score = _cosine_similarity(query_embedding, list(item["embedding"]))
            if score < threshold:
                continue
            results.append(
                RetrievedDocument(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    metadata=metadata,
                    score=score,
                )
            )
        results.sort(key=lambda row: row.score, reverse=True)
        return results[:top_k]

    def delete_documents(self, ids: list[str]) -> None:
        if not ids:
            raise EmptyDocumentListError("Cannot delete an empty id list")
        payload = self._read_payload()
        remove = {doc_id.strip() for doc_id in ids}
        if any(not doc_id for doc_id in remove):
            raise MissingDocumentIdError("Document ids must be non-empty strings")
        payload["documents"] = [
            item for item in payload["documents"] if str(item["id"]) not in remove
        ]
        self._write_payload(payload)

    def delete_collection(self) -> None:
        if not self.collection_exists():
            raise CollectionNotFoundError(
                f"Collection '{self.collection_name}' does not exist"
            )
        try:
            self._store_path.unlink()
        except OSError as exc:
            raise VectorStorePersistenceError(str(exc)) from exc
        self._expected_dimension = None

    def collection_exists(self) -> bool:
        return self._store_path.is_file()

    def count(self) -> int:
        payload = self._read_payload()
        return len(payload["documents"])

    def _validate_config(self) -> None:
        name = self._settings.chroma_collection_name.strip()
        if not name or len(name) < 3:
            raise InvalidVectorStoreConfigError(
                "CHROMA_COLLECTION_NAME must be a non-empty string (≥ 3 chars)"
            )
        persist = self._settings.chroma_persist_directory.strip()
        if not persist:
            raise InvalidVectorStoreConfigError(
                "CHROMA_PERSIST_DIRECTORY must be a non-empty path"
            )

    def _read_payload(self) -> dict[str, Any]:
        if not self.collection_exists():
            raise CollectionNotFoundError(
                f"Collection '{self.collection_name}' does not exist"
            )
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStorePersistenceError(
                f"Failed to read vector store '{self._store_path}': {exc}"
            ) from exc
        if not isinstance(raw, dict) or "documents" not in raw:
            raise VectorStorePersistenceError(
                f"Corrupt vector store file: {self._store_path}"
            )
        return raw

    def _write_payload(self, payload: dict[str, Any]) -> None:
        try:
            self._store_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise VectorStorePersistenceError(
                f"Failed to write vector store '{self._store_path}': {exc}"
            ) from exc


def _resolve_document_id(document: EmbeddedDocument, *, index: int) -> str:
    metadata = document.metadata or {}
    for key in _ID_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    raise MissingDocumentIdError(
        f"Document at index {index} has no id in metadata keys {_ID_METADATA_KEYS}"
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    score = dot / (left_norm * right_norm)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
