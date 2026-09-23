from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from adaptive_rag.domain.models import DenseVectorRecord, EmbeddingIndexSpec
from adaptive_rag.storage.postgres.models import Base, DocumentEmbeddingRow
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore, PgVectorStoreError

DIMENSION = 1024


def _spec(*, model: str = "test-model", expected_records: int = 2) -> EmbeddingIndexSpec:
    return EmbeddingIndexSpec(
        name="test-index",
        provider="test-provider",
        model=model,
        dimension=DIMENSION,
        distance_metric="cosine",
        source_checksum="a" * 64,
        doc_ids_checksum="b" * 64,
        expected_records=expected_records,
        configuration_version="1",
    )


def _record(doc_id: str, axis: int, content_hash: str) -> DenseVectorRecord:
    vector = [0.0] * DIMENSION
    vector[axis] = 1.0
    return DenseVectorRecord(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        chunk_index=0,
        title=f"Title {doc_id}",
        text=f"Text {doc_id}",
        source="unit-test",
        corpus_version="test-v1",
        content_hash=content_hash,
        metadata={"axis": axis},
        embedding=tuple(vector),
    )


def test_offline_session_exercises_metadata_upsert_and_pruning() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, class_=Session, expire_on_commit=False)
    store = PgVectorDenseStore(sessions, index_name="test-index", dimension=DIMENSION)
    first = _record("doc-1", 0, "c" * 64)
    second = _record("doc-2", 1, "d" * 64)

    assert store.prepare(_spec()) is False
    store.upsert([first, second])
    store.upsert([first, second])
    assert store.count() == 2
    assert store.existing_content_hashes() == {
        "doc-1:0": "c" * 64,
        "doc-2:0": "d" * 64,
    }
    assert store.finalize(_spec()).status == "ready"
    assert store.prepare(_spec()) is True

    with sessions() as session:
        stored = session.scalar(
            select(DocumentEmbeddingRow).where(DocumentEmbeddingRow.doc_id == "doc-1")
        )
    assert stored is not None
    assert stored.document_metadata == {"axis": 0}

    with pytest.raises(PgVectorStoreError, match="incompatible"):
        store.prepare(_spec(model="different-model"))

    store.delete_missing(["doc-1:0"])
    assert store.count() == 1
    engine.dispose()


class _FakeSearchSession:
    def __init__(self) -> None:
        self.statement: object | None = None

    def __enter__(self) -> "_FakeSearchSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def scalar(self, statement: object) -> object:
        return SimpleNamespace(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            status="ready",
            actual_records=2,
            expected_records=2,
        )

    def execute(self, statement: object) -> SimpleNamespace:
        self.statement = statement
        return SimpleNamespace(all=lambda: [("doc-1", 0.0), ("doc-2", 0.75)])


class _FakeSearchFactory:
    def __init__(self, session: _FakeSearchSession) -> None:
        self.session = session

    def __call__(self) -> _FakeSearchSession:
        return self.session


def test_search_uses_cosine_distance_and_converts_it_to_similarity() -> None:
    session = _FakeSearchSession()
    factory = cast(sessionmaker[Session], _FakeSearchFactory(session))
    store = PgVectorDenseStore(factory, index_name="test-index", dimension=DIMENSION)

    hits = store.search([1.0, *([0.0] * (DIMENSION - 1))], top_k=2)

    assert [(hit.doc_id, hit.score) for hit in hits] == [
        ("doc-1", 1.0),
        ("doc-2", 0.25),
    ]
    assert session.statement is not None
    sql = str(session.statement.compile(dialect=postgresql.dialect()))  # type: ignore[union-attr]
    assert "<=>" in sql
