"""PostgreSQL/pgvector dense index repository."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID, uuid4

from sqlalchemy import Table, delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.domain.models import (
    DenseSearchHit,
    DenseVectorRecord,
    EmbeddingIndexSnapshot,
    EmbeddingIndexSpec,
)
from adaptive_rag.storage.postgres.models import DocumentEmbeddingRow, EmbeddingIndexRow


class PgVectorStoreError(RuntimeError):
    """Raised when dense index metadata or vector persistence is invalid."""


class PgVectorDenseStore:
    """Own one named, compatibility-checked pgvector index."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        index_name: str,
        dimension: int,
    ) -> None:
        if not index_name.strip() or dimension < 1:
            raise ValueError("index_name and dimension must be valid")
        self._session_factory = session_factory
        self._index_name = index_name
        self._dimension = dimension

    @property
    def index_name(self) -> str:
        return self._index_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def prepare(self, spec: EmbeddingIndexSpec) -> bool:
        self._validate_spec(spec)
        now = datetime.now(UTC)
        try:
            with self._session_factory.begin() as session:
                row = session.scalar(
                    select(EmbeddingIndexRow)
                    .where(EmbeddingIndexRow.name == self._index_name)
                    .with_for_update()
                )
                if row is None:
                    session.add(
                        EmbeddingIndexRow(
                            id=uuid4(),
                            name=spec.name,
                            status="building",
                            provider=spec.provider,
                            model=spec.model,
                            dimension=spec.dimension,
                            distance_metric=spec.distance_metric,
                            source_checksum=spec.source_checksum,
                            doc_ids_checksum=spec.doc_ids_checksum,
                            expected_records=spec.expected_records,
                            actual_records=0,
                            configuration_version=spec.configuration_version,
                            created_at=now,
                            updated_at=now,
                            ready_at=None,
                        )
                    )
                    return False
                self._validate_compatibility(row, spec)
                actual_records = self._count_for_index(session, row.id)
                exact_match = (
                    row.status == "ready"
                    and row.source_checksum == spec.source_checksum
                    and row.doc_ids_checksum == spec.doc_ids_checksum
                    and row.expected_records == spec.expected_records
                    and row.actual_records == spec.expected_records
                    and actual_records == spec.expected_records
                )
                if exact_match:
                    return True
                row.status = "building"
                row.source_checksum = spec.source_checksum
                row.doc_ids_checksum = spec.doc_ids_checksum
                row.expected_records = spec.expected_records
                row.actual_records = actual_records
                row.updated_at = now
                row.ready_at = None
                return False
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to prepare the pgvector index") from error

    def existing_content_hashes(self) -> dict[str, str]:
        try:
            with self._session_factory() as session:
                index_id = self._index_id(session)
                rows = session.execute(
                    select(DocumentEmbeddingRow.chunk_id, DocumentEmbeddingRow.content_hash).where(
                        DocumentEmbeddingRow.index_id == index_id
                    )
                ).all()
                return {str(chunk_id): str(content_hash) for chunk_id, content_hash in rows}
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to inspect existing pgvector records") from error

    def upsert(self, records: Sequence[DenseVectorRecord]) -> None:
        if not records:
            return
        for record in records:
            self._validate_record(record)
        try:
            with self._session_factory.begin() as session:
                index_id = self._index_id(session)
                values = [
                    {
                        "index_id": index_id,
                        "chunk_id": record.chunk_id,
                        "doc_id": record.doc_id,
                        "chunk_index": record.chunk_index,
                        "title": record.title,
                        "text": record.text,
                        "source": record.source,
                        "corpus_version": record.corpus_version,
                        "content_hash": record.content_hash,
                        "metadata": record.metadata,
                        "embedding": list(record.embedding),
                    }
                    for record in records
                ]
                statement = insert(cast(Table, DocumentEmbeddingRow.__table__)).values(values)
                excluded = statement.excluded
                session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["index_id", "chunk_id"],
                        set_={
                            "doc_id": excluded.doc_id,
                            "chunk_index": excluded.chunk_index,
                            "title": excluded.title,
                            "text": excluded.text,
                            "source": excluded.source,
                            "corpus_version": excluded.corpus_version,
                            "content_hash": excluded.content_hash,
                            "metadata": excluded["metadata"],
                            "embedding": excluded.embedding,
                        },
                    )
                )
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to upsert pgvector records") from error

    def delete_missing(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            raise ValueError("chunk_ids cannot be empty")
        try:
            with self._session_factory.begin() as session:
                index_id = self._index_id(session)
                session.execute(
                    delete(DocumentEmbeddingRow).where(
                        DocumentEmbeddingRow.index_id == index_id,
                        DocumentEmbeddingRow.chunk_id.not_in(list(chunk_ids)),
                    )
                )
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to remove stale pgvector records") from error

    def finalize(self, spec: EmbeddingIndexSpec) -> EmbeddingIndexSnapshot:
        self._validate_spec(spec)
        now = datetime.now(UTC)
        try:
            with self._session_factory.begin() as session:
                row = self._index_row(session, for_update=True)
                self._validate_compatibility(row, spec)
                actual_records = self._count_for_index(session, row.id)
                row.actual_records = actual_records
                row.updated_at = now
                if actual_records != spec.expected_records:
                    raise PgVectorStoreError(
                        f"pgvector index has {actual_records} records; "
                        f"expected {spec.expected_records}"
                    )
                row.status = "ready"
                row.ready_at = now
                snapshot = self._to_snapshot(row)
            return snapshot
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to finalize the pgvector index") from error

    def snapshot(self) -> EmbeddingIndexSnapshot:
        try:
            with self._session_factory() as session:
                return self._to_snapshot(self._index_row(session))
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to read pgvector index metadata") from error

    def count(self) -> int:
        try:
            with self._session_factory() as session:
                return self._count_for_index(session, self._index_id(session))
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to count pgvector records") from error

    def search(self, vector: Sequence[float], top_k: int) -> list[DenseSearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self._validate_vector(vector)
        try:
            with self._session_factory() as session:
                index = self._index_row(session)
                if index.status != "ready" or index.actual_records != index.expected_records:
                    raise PgVectorStoreError("pgvector index is not ready")
                distance = DocumentEmbeddingRow.embedding.cosine_distance(list(vector)).label(
                    "distance"
                )
                rows = session.execute(
                    select(DocumentEmbeddingRow.doc_id, distance)
                    .where(DocumentEmbeddingRow.index_id == index.id)
                    .order_by(distance, DocumentEmbeddingRow.doc_id)
                    .limit(top_k)
                ).all()
        except SQLAlchemyError as error:
            raise PgVectorStoreError("Unable to search the pgvector index") from error
        hits: list[DenseSearchHit] = []
        for doc_id, raw_distance in rows:
            score = 1.0 - float(raw_distance)
            if not math.isfinite(score):
                raise PgVectorStoreError("pgvector returned a non-finite distance")
            hits.append(DenseSearchHit(doc_id=str(doc_id), score=score))
        return hits

    def close(self) -> None:
        """The shared Engine owns connections; the store has nothing separate to close."""

    def _validate_spec(self, spec: EmbeddingIndexSpec) -> None:
        if spec.name != self._index_name or spec.dimension != self._dimension:
            raise PgVectorStoreError("Dense index spec does not match the connected store")
        if spec.distance_metric != "cosine":
            raise PgVectorStoreError("Only cosine pgvector indexes are supported")

    @staticmethod
    def _validate_compatibility(row: EmbeddingIndexRow, spec: EmbeddingIndexSpec) -> None:
        actual = (
            row.provider,
            row.model,
            row.dimension,
            row.distance_metric,
            row.configuration_version,
        )
        expected = (
            spec.provider,
            spec.model,
            spec.dimension,
            spec.distance_metric,
            spec.configuration_version,
        )
        if actual != expected:
            raise PgVectorStoreError(
                "Existing dense index has incompatible provider, model, dimension, metric, "
                "or configuration version; create a new index name"
            )

    def _validate_record(self, record: DenseVectorRecord) -> None:
        if not record.chunk_id or not record.doc_id or record.chunk_index < 0:
            raise PgVectorStoreError("Dense vector record identifiers are invalid")
        self._validate_vector(record.embedding)

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise PgVectorStoreError(
                f"Dense vector dimension {len(vector)} does not match {self._dimension}"
            )
        if not all(math.isfinite(float(value)) for value in vector):
            raise PgVectorStoreError("Dense vector contains a non-finite value")

    def _index_row(self, session: Session, *, for_update: bool = False) -> EmbeddingIndexRow:
        statement = select(EmbeddingIndexRow).where(
            EmbeddingIndexRow.name == self._index_name
        )
        if for_update:
            statement = statement.with_for_update()
        row = session.scalar(statement)
        if row is None:
            raise PgVectorStoreError(f"Dense index {self._index_name!r} does not exist")
        return row

    def _index_id(self, session: Session) -> UUID:
        return self._index_row(session).id

    @staticmethod
    def _count_for_index(session: Session, index_id: UUID) -> int:
        return int(
            session.scalar(
                select(func.count()).select_from(DocumentEmbeddingRow).where(
                    DocumentEmbeddingRow.index_id == index_id
                )
            )
            or 0
        )

    @staticmethod
    def _to_snapshot(row: EmbeddingIndexRow) -> EmbeddingIndexSnapshot:
        return EmbeddingIndexSnapshot(
            name=row.name,
            status=cast(Literal["building", "ready", "failed"], row.status),
            provider=row.provider,
            model=row.model,
            dimension=row.dimension,
            distance_metric=cast(Literal["cosine"], row.distance_metric),
            source_checksum=row.source_checksum,
            doc_ids_checksum=row.doc_ids_checksum,
            expected_records=row.expected_records,
            actual_records=row.actual_records,
            configuration_version=row.configuration_version,
            updated_at=row.updated_at,
        )
