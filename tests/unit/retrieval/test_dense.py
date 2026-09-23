from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import (
    DenseSearchHit,
    DenseVectorRecord,
    EmbeddingIndexSnapshot,
    EmbeddingIndexSpec,
)
from adaptive_rag.retrieval.dense import (
    DenseConfig,
    DenseIndexError,
    DenseRetriever,
    build_dense_index,
    load_dense_config,
    verify_dense_index,
)

CORPUS_CHECKSUM = "b" * 64
DIMENSION = 1024


class FakeEmbedder:
    provider = "test-provider"
    model_name = "test/model"
    dimension = DIMENSION

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [[float(index), *([0.0] * (DIMENSION - 1))] for index, _ in enumerate(texts)]

    def encode_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.0] * DIMENSION


class FakeStore:
    index_name = "test_dense"
    dimension = DIMENSION

    def __init__(self) -> None:
        self.records: dict[str, DenseVectorRecord] = {}
        self.spec: EmbeddingIndexSpec | None = None
        self.ready = False
        self.hits = [DenseSearchHit("d2", 0.8), DenseSearchHit("d1", 0.7)]

    def prepare(self, spec: EmbeddingIndexSpec) -> bool:
        self.spec = spec
        return self.ready and len(self.records) == spec.expected_records

    def existing_content_hashes(self) -> dict[str, str]:
        return {key: value.content_hash for key, value in self.records.items()}

    def upsert(self, records: list[DenseVectorRecord]) -> None:
        self.records.update({record.chunk_id: record for record in records})

    def delete_missing(self, chunk_ids: list[str]) -> None:
        self.records = {key: value for key, value in self.records.items() if key in chunk_ids}

    def finalize(self, spec: EmbeddingIndexSpec) -> EmbeddingIndexSnapshot:
        self.spec = spec
        self.ready = True
        return self.snapshot()

    def snapshot(self) -> EmbeddingIndexSnapshot:
        if self.spec is None:
            raise RuntimeError("not prepared")
        return EmbeddingIndexSnapshot(
            name=self.spec.name,
            status="ready" if self.ready else "building",
            provider=self.spec.provider,
            model=self.spec.model,
            dimension=self.spec.dimension,
            distance_metric=self.spec.distance_metric,
            source_checksum=self.spec.source_checksum,
            doc_ids_checksum=self.spec.doc_ids_checksum,
            expected_records=self.spec.expected_records,
            actual_records=len(self.records),
            configuration_version=self.spec.configuration_version,
            updated_at=datetime.now(UTC),
        )

    def count(self) -> int:
        return len(self.records)

    def search(self, vector: list[float], top_k: int) -> list[DenseSearchHit]:
        assert len(vector) == DIMENSION
        return self.hits[:top_k]


def _config() -> DenseConfig:
    return DenseConfig(
        name="test-dense",
        provider="test-provider",
        model_name="test/model",
        dimension=DIMENSION,
        top_k=2,
        database_batch_size=2,
        index_name="test_dense",
        configuration_version="1",
    )


def _corpus() -> dict[str, CorpusDocument]:
    return {
        "d2": CorpusDocument("d2", "Title two", "Body two", {"year": 2024}),
        "d1": CorpusDocument("d1", "Title one", "Body one", {}),
        "d3": CorpusDocument("d3", "", "Body three", {}),
    }


def test_builds_stably_and_reuses_matching_index() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()

    snapshot = build_dense_index(
        _corpus(),
        corpus_checksum=CORPUS_CHECKSUM,
        config=_config(),
        embedder=embedder,
        store=store,
    )
    reused = build_dense_index(
        _corpus(),
        corpus_checksum=CORPUS_CHECKSUM,
        config=_config(),
        embedder=embedder,
        store=store,
    )

    assert list(record.doc_id for record in store.records.values()) == ["d1", "d2", "d3"]
    assert [len(batch) for batch in embedder.document_batches] == [2, 1]
    assert snapshot.actual_records == 3
    assert reused.status == "ready"


def test_content_change_embeds_only_changed_document() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    corpus = _corpus()
    build_dense_index(
        corpus,
        corpus_checksum=CORPUS_CHECKSUM,
        config=_config(),
        embedder=embedder,
        store=store,
    )
    store.ready = False
    changed = {**corpus, "d2": replace(corpus["d2"], text="Changed body")}

    build_dense_index(
        changed,
        corpus_checksum="c" * 64,
        config=_config(),
        embedder=embedder,
        store=store,
    )

    assert embedder.document_batches[-1] == ["Title two\n\nChanged body"]


def test_retriever_and_verification_use_database_snapshot() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()
    build_dense_index(
        _corpus(),
        corpus_checksum=CORPUS_CHECKSUM,
        config=_config(),
        embedder=embedder,
        store=store,
    )
    verify_dense_index(
        corpus=_corpus(),
        corpus_checksum=CORPUS_CHECKSUM,
        config=_config(),
        embedder=embedder,
        store=store,
    )

    results = DenseRetriever(embedder=embedder, store=store, config=_config()).retrieve("q", 2)

    assert [result.doc_id for result in results] == ["d2", "d1"]
    assert embedder.queries == ["q"]


def test_rejects_incompatible_provider() -> None:
    embedder = FakeEmbedder()
    embedder.provider = "other"
    with pytest.raises(DenseIndexError, match="provider"):
        build_dense_index(
            _corpus(),
            corpus_checksum=CORPUS_CHECKSUM,
            config=_config(),
            embedder=embedder,
            store=FakeStore(),
        )


def test_load_dense_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "dense.yaml"
    path.write_text("name: invalid\nstrategy: hybrid\nmodel_name: test/model\n")
    with pytest.raises(DenseIndexError, match="Unable to load dense config"):
        load_dense_config(path)
