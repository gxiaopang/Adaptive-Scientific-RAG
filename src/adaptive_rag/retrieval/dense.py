"""Online embedding and PostgreSQL/pgvector dense retrieval orchestration."""

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import (
    DenseVectorRecord,
    EmbeddingIndexSnapshot,
    EmbeddingIndexSpec,
    ScoredDocument,
)
from adaptive_rag.domain.protocols import DenseVectorStore, EmbeddingProvider


class DenseIndexError(RuntimeError):
    """Raised when dense configuration, indexing, or metadata validation fails."""


class DenseConfig(BaseModel):
    """Frozen provider and pgvector parameters for the SciFact dense index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    strategy: Literal["dense"] = "dense"
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    dimension: Literal[1024] = 1024
    distance_metric: Literal["cosine"] = "cosine"
    top_k: int = Field(gt=0)
    database_batch_size: int = Field(gt=0)
    index_name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    configuration_version: str = Field(min_length=1, max_length=64)


class DenseRetriever:
    """Encode one query and map pgvector hits to the shared result model."""

    strategy = "dense"

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        store: DenseVectorStore,
        config: DenseConfig,
    ) -> None:
        _validate_components(config, embedder, store)
        snapshot = store.snapshot()
        _validate_snapshot(snapshot, config)
        if snapshot.status != "ready" or snapshot.actual_records != snapshot.expected_records:
            raise DenseIndexError("Dense index is not ready")
        self._embedder = embedder
        self._store = store
        self.config = config

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        hits = self._store.search(self._embedder.encode_query(query), top_k)
        return [
            ScoredDocument(doc_id=hit.doc_id, score=hit.score, rank=rank)
            for rank, hit in enumerate(hits, start=1)
        ]


def build_dense_index(
    corpus: Mapping[str, CorpusDocument],
    *,
    corpus_checksum: str,
    config: DenseConfig,
    embedder: EmbeddingProvider,
    store: DenseVectorStore,
) -> EmbeddingIndexSnapshot:
    """Build or reuse an idempotent index, embedding only new or changed documents."""

    if not corpus:
        raise ValueError("corpus cannot be empty")
    _validate_components(config, embedder, store)
    ordered_documents = sorted(corpus.values(), key=lambda document: document.doc_id)
    spec = EmbeddingIndexSpec(
        name=config.index_name,
        provider=config.provider,
        model=config.model_name,
        dimension=config.dimension,
        distance_metric=config.distance_metric,
        source_checksum=corpus_checksum,
        doc_ids_checksum=doc_ids_checksum([document.doc_id for document in ordered_documents]),
        expected_records=len(ordered_documents),
        configuration_version=config.configuration_version,
    )
    if store.prepare(spec):
        snapshot = store.snapshot()
        _validate_snapshot(snapshot, config)
        return snapshot

    existing_hashes = store.existing_content_hashes()
    prepared: list[tuple[CorpusDocument, str, str]] = []
    for document in ordered_documents:
        chunk_id = f"scifact:{document.doc_id}:0"
        content_hash = hashlib.sha256(document.retrieval_text.encode()).hexdigest()
        if existing_hashes.get(chunk_id) != content_hash:
            prepared.append((document, chunk_id, content_hash))

    for start in range(0, len(prepared), config.database_batch_size):
        batch = prepared[start : start + config.database_batch_size]
        embeddings = embedder.encode_documents(
            [document.retrieval_text for document, _, _ in batch]
        )
        if len(embeddings) != len(batch):
            raise DenseIndexError("Embedding provider returned the wrong vector count")
        store.upsert(
            [
                _to_vector_record(
                    document,
                    chunk_id=chunk_id,
                    content_hash=content_hash,
                    embedding=embedding,
                    corpus_version=corpus_checksum,
                )
                for (document, chunk_id, content_hash), embedding in zip(
                    batch, embeddings, strict=True
                )
            ]
        )
    store.delete_missing([f"scifact:{document.doc_id}:0" for document in ordered_documents])
    return store.finalize(spec)


def verify_dense_index(
    *,
    corpus: Mapping[str, CorpusDocument],
    corpus_checksum: str,
    config: DenseConfig,
    embedder: EmbeddingProvider,
    store: DenseVectorStore,
) -> EmbeddingIndexSnapshot:
    """Fail closed if database metadata, provider configuration, or corpus diverges."""

    _validate_components(config, embedder, store)
    snapshot = store.snapshot()
    expected_doc_ids_checksum = doc_ids_checksum(sorted(corpus))
    if snapshot.source_checksum != corpus_checksum:
        raise DenseIndexError("Dense index corpus checksum does not match the current dataset")
    if snapshot.doc_ids_checksum != expected_doc_ids_checksum:
        raise DenseIndexError("Dense index document IDs do not match the current dataset")
    if snapshot.expected_records != len(corpus) or snapshot.actual_records != len(corpus):
        raise DenseIndexError("Dense index record count does not match the current dataset")
    if snapshot.status != "ready":
        raise DenseIndexError("Dense index is not ready")
    _validate_snapshot(snapshot, config)
    return snapshot


def load_dense_config(path: Path) -> DenseConfig:
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return DenseConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise DenseIndexError(f"Unable to load dense config from {path}: {error}") from error


def doc_ids_checksum(doc_ids: Sequence[str]) -> str:
    serialized_ids = "".join(f"{doc_id}\n" for doc_id in doc_ids)
    return hashlib.sha256(serialized_ids.encode()).hexdigest()


def _validate_components(
    config: DenseConfig,
    embedder: EmbeddingProvider,
    store: DenseVectorStore,
) -> None:
    if embedder.dimension != config.dimension or store.dimension != config.dimension:
        raise DenseIndexError("Configured, provider, and pgvector dimensions must match")
    if embedder.model_name != config.model_name or embedder.provider != config.provider:
        raise DenseIndexError("Configured embedding provider and loaded provider must match")
    if store.index_name != config.index_name:
        raise DenseIndexError("Configured and connected pgvector index names must match")


def _validate_snapshot(snapshot: EmbeddingIndexSnapshot, config: DenseConfig) -> None:
    identity = (
        snapshot.name,
        snapshot.provider,
        snapshot.model,
        snapshot.dimension,
        snapshot.distance_metric,
        snapshot.configuration_version,
    )
    expected = (
        config.index_name,
        config.provider,
        config.model_name,
        config.dimension,
        config.distance_metric,
        config.configuration_version,
    )
    if identity != expected:
        raise DenseIndexError("Dense index metadata is incompatible with configuration")


def _to_vector_record(
    document: CorpusDocument,
    *,
    chunk_id: str,
    content_hash: str,
    embedding: Sequence[float],
    corpus_version: str,
) -> DenseVectorRecord:
    return DenseVectorRecord(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        chunk_index=0,
        title=document.title,
        text=document.text,
        source="scifact",
        corpus_version=corpus_version,
        content_hash=content_hash,
        metadata=document.metadata,
        embedding=tuple(embedding),
    )
