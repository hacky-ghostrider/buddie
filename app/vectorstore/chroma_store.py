"""ChromaDB vector store Strategy.

Persists ``EmbeddedDocument`` vectors locally via Chroma and runs cosine-space
similarity search. Embedding computation and LLM generation stay elsewhere.

Chroma client construction is injectable so unit tests can use a disk-backed
double without depending on native Chroma bindings (which can segfault on
some Windows hosts during ``collection.add``).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

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
    VectorStoreError,
    VectorStorePersistenceError,
)
from app.vectorstore.models import AddDocumentsResult, VectorDocument

logger = logging.getLogger(__name__)

# Preferred metadata keys used to resolve a stable document id.
_ID_METADATA_KEYS: tuple[str, ...] = (
    MetadataKeys.CHUNK_ID,
    "id",
    "document_id",
)


class _ChromaCollection(Protocol):
    """Minimal Chroma collection surface used by this Strategy."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> Any: ...

    def delete(self, ids: list[str]) -> Any: ...

    def count(self) -> int: ...

    def get(self, ids: list[str] | None = None, include: list[str] | None = None) -> Any: ...


class _ChromaClient(Protocol):
    """Minimal Chroma client surface used by this Strategy."""

    def create_collection(
        self,
        name: str,
        *,
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None,
        get_or_create: bool = False,
    ) -> _ChromaCollection: ...

    def get_collection(
        self,
        name: str,
        *,
        embedding_function: Any = None,
    ) -> _ChromaCollection: ...

    def delete_collection(self, name: str) -> None: ...

    def list_collections(self) -> Sequence[Any]: ...


ClientFactory = Callable[[str], _ChromaClient]


def _default_client_factory(persist_directory: str) -> _ChromaClient:
    """Build a persistent Chroma client for ``persist_directory``.

    On Windows, prefers the pure-Python ``SegmentAPI`` when ``hnswlib`` is
    importable (avoids known Rust-binding access violations). Otherwise falls
    back to ``PersistentClient``.

    Args:
        persist_directory: Local directory for Chroma persistence files.

    Returns:
        A Chroma client bound to the given directory.

    Raises:
        VectorStorePersistenceError: Client construction failed.
    """
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except Exception as exc:  # noqa: BLE001
        raise VectorStorePersistenceError(
            f"chromadb is unavailable: {exc}"
        ) from exc

    path = str(Path(persist_directory))
    Path(path).mkdir(parents=True, exist_ok=True)

    try:
        if sys.platform == "win32" and _hnswlib_available():
            logger.info(
                "Using Chroma SegmentAPI on Windows: persist_directory=%s",
                path,
            )
            return chromadb.Client(
                ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=path,
                    chroma_api_impl="chromadb.api.segment.SegmentAPI",
                )
            )

        logger.info("Using Chroma PersistentClient: persist_directory=%s", path)
        return chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except VectorStoreError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VectorStorePersistenceError(
            f"Failed to open Chroma persist directory '{path}': {exc}"
        ) from exc


def _hnswlib_available() -> bool:
    """Return True when hnswlib can be imported (SegmentAPI dependency)."""
    try:
        import hnswlib  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class ChromaVectorStore(VectorStore):
    """Persist embeddings in a local Chroma collection.

    Args:
        settings: Application settings (collection name + persist path).
        client_factory: Optional factory for constructing the Chroma client.
            Inject a fake in unit / persistence tests.
        client: Optional pre-built client (skips factory).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client_factory: ClientFactory | None = None,
        client: _ChromaClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory or _default_client_factory
        self._client = client
        self._expected_dimension: int | None = None
        self._validate_config()

    @property
    def collection_name(self) -> str:
        """Return the configured Chroma collection name."""
        return self._settings.chroma_collection_name

    @property
    def persist_directory(self) -> str:
        """Return the configured persistence directory."""
        return self._settings.chroma_persist_directory

    def create_collection(self) -> None:
        """Create the configured collection.

        Raises:
            CollectionAlreadyExistsError: Collection already present.
            VectorStorePersistenceError: Backend create failed.
        """
        name = self.collection_name
        if self.collection_exists():
            raise CollectionAlreadyExistsError(
                f"Collection '{name}' already exists"
            )

        client = self._get_client()
        logger.info(
            "Collection creation started: name=%s persist_directory=%s",
            name,
            self.persist_directory,
        )
        try:
            client.create_collection(
                name=name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
        except CollectionAlreadyExistsError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=name)
            if mapped is not None:
                raise mapped from exc
            logger.error(
                "Collection creation failed: name=%s error=%s",
                name,
                exc,
            )
            raise VectorStorePersistenceError(
                f"Failed to create collection '{name}': {exc}"
            ) from exc

        logger.info("Collection creation finished: name=%s", name)

    def add_documents(self, documents: list[EmbeddedDocument]) -> AddDocumentsResult:
        """Validate and persist embedded documents.

        Document ids are resolved from metadata (``chunk_id``, then ``id``,
        then ``document_id``). Embedding dimensions must be consistent across
        the batch and with any previously stored vectors in this process.
        """
        if not documents:
            raise EmptyDocumentListError("Cannot add an empty document list")

        collection = self._require_collection()
        records = [_to_vector_document(doc, index=i) for i, doc in enumerate(documents)]
        self._validate_dimensions(records)
        self._reject_duplicate_ids_in_batch(records)
        self._reject_existing_ids(collection, [record.id for record in records])

        ids = [record.id for record in records]
        embeddings = [record.embedding for record in records]
        texts = [record.text for record in records]
        metadatas = [_sanitize_metadata(record.metadata) for record in records]

        logger.info(
            "Documents add started: count=%s collection=%s persist_directory=%s",
            len(records),
            self.collection_name,
            self.persist_directory,
        )
        try:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=self.collection_name)
            if mapped is not None:
                raise mapped from exc
            logger.error(
                "Documents add failed: collection=%s error=%s",
                self.collection_name,
                exc,
            )
            raise VectorStorePersistenceError(
                f"Failed to add documents to '{self.collection_name}': {exc}"
            ) from exc

        total = self._safe_count(collection)
        logger.info(
            "Documents add finished: added=%s collection=%s total_vectors=%s",
            len(ids),
            self.collection_name,
            total,
        )
        return AddDocumentsResult(
            ids=ids,
            added_count=len(ids),
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
        """Return nearest neighbors for ``query_embedding`` (cosine similarity).

        Chroma cosine space returns **distance** ``≈ 1 - cosine_similarity``.
        This method converts distance to a ``[0, 1]`` similarity score and
        keeps rows with ``score >= score_threshold``, best-first.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidVectorStoreConfigError(
                f"top_k must be a positive integer, got {top_k!r}"
            )
        try:
            threshold = float(score_threshold)
        except (TypeError, ValueError) as exc:
            raise InvalidVectorStoreConfigError(
                f"score_threshold must be a float in [0, 1], got {score_threshold!r}"
            ) from exc
        if threshold < 0.0 or threshold > 1.0:
            raise InvalidVectorStoreConfigError(
                f"score_threshold must be between 0 and 1 inclusive, got {threshold}"
            )
        if not query_embedding:
            raise InvalidEmbeddingDimensionError("query_embedding must be non-empty")

        if (
            self._expected_dimension is not None
            and len(query_embedding) != self._expected_dimension
        ):
            raise InvalidEmbeddingDimensionError(
                f"Query embedding dimension {len(query_embedding)} does not match "
                f"collection dimension {self._expected_dimension}"
            )

        collection = self._require_collection()
        stored = self._safe_count_int(collection)
        if stored == 0:
            logger.info(
                "Similarity search finished: collection=%s retrieved_count=0 "
                "(empty collection)",
                self.collection_name,
            )
            return []

        n_results = min(top_k, stored)

        logger.info(
            "Similarity search started: collection=%s top_k=%s score_threshold=%s "
            "query_dim=%s metadata_filter=%s",
            self.collection_name,
            top_k,
            threshold,
            len(query_embedding),
            metadata_filter is not None,
        )

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [list(query_embedding)],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        try:
            raw = collection.query(**query_kwargs)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=self.collection_name)
            if mapped is not None:
                raise mapped from exc
            logger.error(
                "Similarity search failed: collection=%s error=%s",
                self.collection_name,
                exc,
            )
            raise VectorStorePersistenceError(
                f"Similarity search failed in '{self.collection_name}': {exc}"
            ) from exc

        results = _parse_query_results(raw, score_threshold=threshold)
        # Cap at top_k after threshold (Chroma may return n_results before filter).
        results = results[:top_k]

        avg_score = (
            sum(item.score for item in results) / len(results) if results else 0.0
        )
        logger.info(
            "Similarity search finished: collection=%s retrieved_count=%s "
            "average_score=%.4f",
            self.collection_name,
            len(results),
            avg_score,
        )
        return results

    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by id."""
        if not ids:
            raise EmptyDocumentListError("Cannot delete an empty id list")

        cleaned = [doc_id.strip() for doc_id in ids]
        if any(not doc_id for doc_id in cleaned):
            raise MissingDocumentIdError("Document ids must be non-empty strings")

        collection = self._require_collection()
        logger.info(
            "Documents delete started: count=%s collection=%s",
            len(cleaned),
            self.collection_name,
        )
        try:
            collection.delete(ids=cleaned)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=self.collection_name)
            if mapped is not None:
                raise mapped from exc
            raise VectorStorePersistenceError(
                f"Failed to delete documents from '{self.collection_name}': {exc}"
            ) from exc

        logger.info(
            "Documents delete finished: count=%s collection=%s remaining=%s",
            len(cleaned),
            self.collection_name,
            self._safe_count(collection),
        )

    def delete_collection(self) -> None:
        """Delete the configured collection entirely."""
        name = self.collection_name
        if not self.collection_exists():
            raise CollectionNotFoundError(f"Collection '{name}' does not exist")

        client = self._get_client()
        logger.info("Collection delete started: name=%s", name)
        try:
            client.delete_collection(name)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=name)
            if mapped is not None:
                raise mapped from exc
            raise VectorStorePersistenceError(
                f"Failed to delete collection '{name}': {exc}"
            ) from exc

        self._expected_dimension = None
        logger.info("Collection delete finished: name=%s", name)

    def collection_exists(self) -> bool:
        """Return whether the configured collection exists."""
        name = self.collection_name
        client = self._get_client()
        try:
            for collection in client.list_collections():
                if _collection_name(collection) == name:
                    return True
            return False
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStorePersistenceError(
                f"Failed to list collections in '{self.persist_directory}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of stored vectors in the collection."""
        collection = self._require_collection()
        try:
            value = int(collection.count())
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStorePersistenceError(
                f"Failed to count vectors in '{self.collection_name}': {exc}"
            ) from exc
        logger.info(
            "Collection count: name=%s vectors=%s",
            self.collection_name,
            value,
        )
        return value

    def _validate_config(self) -> None:
        """Validate vector-store settings before use."""
        vector_db = self._settings.vector_db.strip().lower()
        if vector_db != "chroma":
            raise InvalidVectorStoreConfigError(
                f"VECTOR_DB must be 'chroma' for ChromaVectorStore, got '{self._settings.vector_db}'"
            )

        name = self._settings.chroma_collection_name.strip()
        if not name:
            raise InvalidVectorStoreConfigError(
                "CHROMA_COLLECTION_NAME must be a non-empty string"
            )
        if len(name) < 3:
            raise InvalidVectorStoreConfigError(
                "CHROMA_COLLECTION_NAME must be at least 3 characters"
            )

        persist = self._settings.chroma_persist_directory.strip()
        if not persist:
            raise InvalidVectorStoreConfigError(
                "CHROMA_PERSIST_DIRECTORY must be a non-empty path"
            )

    def _get_client(self) -> _ChromaClient:
        """Lazy-open and cache the Chroma client."""
        if self._client is not None:
            return self._client

        directory = self.persist_directory
        logger.info(
            "Opening vector store client: vector_db=%s persist_directory=%s",
            self._settings.vector_db,
            directory,
        )
        try:
            self._client = self._client_factory(directory)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Vector store client open failed: persist_directory=%s error=%s",
                directory,
                exc,
            )
            raise VectorStorePersistenceError(
                f"Failed to open vector store at '{directory}': {exc}"
            ) from exc
        return self._client

    def _require_collection(self) -> _ChromaCollection:
        """Return the configured collection or raise CollectionNotFoundError."""
        name = self.collection_name
        if not self.collection_exists():
            raise CollectionNotFoundError(f"Collection '{name}' does not exist")
        client = self._get_client()
        try:
            return client.get_collection(name=name, embedding_function=None)
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            mapped = _map_chroma_error(exc, collection_name=name)
            if mapped is not None:
                raise mapped from exc
            raise VectorStorePersistenceError(
                f"Failed to open collection '{name}': {exc}"
            ) from exc

    def _validate_dimensions(self, records: Sequence[VectorDocument]) -> None:
        """Ensure all vectors share one positive dimension."""
        dimensions = {record.dimension for record in records}
        if len(dimensions) != 1:
            raise InvalidEmbeddingDimensionError(
                f"Inconsistent embedding dimensions in batch: {sorted(dimensions)}"
            )
        dimension = next(iter(dimensions))
        if dimension <= 0:
            raise InvalidEmbeddingDimensionError(
                f"Embedding dimension must be positive, got {dimension}"
            )
        if self._expected_dimension is None:
            self._expected_dimension = dimension
            return
        if dimension != self._expected_dimension:
            raise InvalidEmbeddingDimensionError(
                f"Embedding dimension {dimension} does not match "
                f"collection dimension {self._expected_dimension}"
            )

    def _reject_duplicate_ids_in_batch(self, records: Sequence[VectorDocument]) -> None:
        """Reject duplicate ids within a single add batch."""
        seen: set[str] = set()
        for record in records:
            if record.id in seen:
                raise DuplicateDocumentIdError(
                    f"Duplicate document id in batch: '{record.id}'"
                )
            seen.add(record.id)

    def _reject_existing_ids(
        self,
        collection: _ChromaCollection,
        ids: Sequence[str],
    ) -> None:
        """Reject ids that already exist in the collection."""
        try:
            existing = collection.get(ids=list(ids), include=[])
        except Exception as exc:  # noqa: BLE001
            # Some fakes / older clients may not support get; fall through to add.
            logger.debug("Pre-add id check skipped: error=%s", exc)
            return

        existing_ids = list(existing.get("ids", []) if isinstance(existing, Mapping) else [])
        if existing_ids:
            raise DuplicateDocumentIdError(
                f"Document id(s) already exist: {existing_ids}"
            )

    def _safe_count(self, collection: _ChromaCollection) -> int | str:
        """Best-effort count for logging (never raises)."""
        try:
            return int(collection.count())
        except Exception:  # noqa: BLE001
            return "unknown"

    def _safe_count_int(self, collection: _ChromaCollection) -> int:
        """Best-effort integer count (``0`` when unavailable)."""
        value = self._safe_count(collection)
        return value if isinstance(value, int) else 0


def _to_vector_document(document: EmbeddedDocument, *, index: int) -> VectorDocument:
    """Convert an ``EmbeddedDocument`` into a storeable ``VectorDocument``."""
    doc_id = _resolve_document_id(document, index=index)
    return VectorDocument(
        id=doc_id,
        text=document.text,
        embedding=list(document.embedding),
        metadata=dict(document.metadata),
    )


def _resolve_document_id(document: EmbeddedDocument, *, index: int) -> str:
    """Resolve a stable id from document metadata.

    Args:
        document: Embedded document with metadata.
        index: Position in the batch (used only in error messages).

    Returns:
        Non-empty document id string.

    Raises:
        MissingDocumentIdError: No usable id key was found.
    """
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


def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce metadata values into Chroma-safe scalar types.

    Chroma accepts str / int / float / bool. Other values are stringified so
    provenance is preserved without nested structures.
    """
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def _collection_name(collection: Any) -> str:
    """Extract a collection name from a Chroma collection object or string."""
    if isinstance(collection, str):
        return collection
    name = getattr(collection, "name", None)
    if isinstance(name, str):
        return name
    return str(collection)


def _parse_query_results(
    raw: Any,
    *,
    score_threshold: float,
) -> list[RetrievedDocument]:
    """Convert a Chroma ``query`` payload into scored ``RetrievedDocument`` rows.

    Args:
        raw: Chroma query response (nested lists per query).
        score_threshold: Minimum cosine-derived similarity to keep.

    Returns:
        Matches sorted by descending score.
    """
    if not isinstance(raw, Mapping):
        return []

    ids_batch = list(raw.get("ids") or [[]])
    docs_batch = list(raw.get("documents") or [[]])
    metas_batch = list(raw.get("metadatas") or [[]])
    dists_batch = list(raw.get("distances") or [[]])

    ids = list(ids_batch[0]) if ids_batch else []
    documents = list(docs_batch[0]) if docs_batch else []
    metadatas = list(metas_batch[0]) if metas_batch else []
    distances = list(dists_batch[0]) if dists_batch else []

    # Pad shorter parallel lists so zip stays aligned.
    length = len(ids)
    while len(documents) < length:
        documents.append("")
    while len(metadatas) < length:
        metadatas.append({})
    while len(distances) < length:
        distances.append(1.0)

    results: list[RetrievedDocument] = []
    for doc_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances, strict=False
    ):
        if doc_id is None:
            continue
        id_text = str(doc_id).strip()
        body = "" if text is None else str(text)
        if not id_text or not body.strip():
            continue
        score = _distance_to_similarity(distance)
        if score < score_threshold:
            continue
        meta = dict(metadata) if isinstance(metadata, Mapping) else {}
        results.append(
            RetrievedDocument(
                id=id_text,
                text=body,
                metadata=meta,
                score=score,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _distance_to_similarity(distance: Any) -> float:
    """Convert Chroma cosine **distance** to a ``[0, 1]`` similarity score.

    For ``hnsw:space = cosine``, Chroma distance is ``1 - cosine_similarity``.
    """
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    score = 1.0 - value
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _map_chroma_error(
    exc: Exception,
    *,
    collection_name: str,
) -> VectorStoreError | None:
    """Map known Chroma exception types onto domain errors."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()

    if "unique" in name or "already exists" in message or "duplicate" in name:
        if "collection" in message:
            return CollectionAlreadyExistsError(
                f"Collection '{collection_name}' already exists"
            )
        return DuplicateDocumentIdError(str(exc))

    if "notfound" in name or "does not exist" in message or "not found" in message:
        return CollectionNotFoundError(f"Collection '{collection_name}' does not exist")

    if "dimension" in name or "dimension" in message:
        return InvalidEmbeddingDimensionError(str(exc))

    if isinstance(exc, (OSError, PermissionError)):
        return VectorStorePersistenceError(str(exc))

    return None
