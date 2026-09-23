from pathlib import Path
from threading import Barrier

import pytest

from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.retrieval.hybrid import (
    HybridConfig,
    HybridRetrievalError,
    HybridRetriever,
    load_hybrid_config,
    reciprocal_rank_fusion,
)


class StaticRetriever:
    def __init__(self, strategy: str, results: list[ScoredDocument]) -> None:
        self._strategy = strategy
        self.results = results
        self.calls: list[tuple[str, int]] = []

    @property
    def strategy(self) -> str:
        return self._strategy

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class BarrierRetriever(StaticRetriever):
    def __init__(
        self,
        strategy: str,
        results: list[ScoredDocument],
        barrier: Barrier,
    ) -> None:
        super().__init__(strategy, results)
        self.barrier = barrier

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        self.barrier.wait(timeout=1)
        return super().retrieve(query, top_k)


def _document(doc_id: str, score: float, rank: int) -> ScoredDocument:
    return ScoredDocument(doc_id=doc_id, score=score, rank=rank)


def _config() -> HybridConfig:
    return HybridConfig(
        name="test-hybrid",
        bm25_candidate_k=3,
        dense_candidate_k=2,
        rrf_k=60,
        top_k=2,
    )


def test_rrf_accumulates_overlap_and_uses_only_ranks() -> None:
    bm25 = [_document("shared", 100.0, 1), _document("sparse", 99.0, 2)]
    dense = [_document("dense", -5.0, 1), _document("shared", -100.0, 2)]

    results = reciprocal_rank_fusion((bm25, dense), rrf_k=60, top_k=3)

    assert [result.doc_id for result in results] == ["shared", "dense", "sparse"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(1 / 61)
    assert [result.rank for result in results] == [1, 2, 3]


def test_rrf_is_invariant_to_raw_score_scale() -> None:
    first = (
        [_document("a", 0.1, 1), _document("b", 0.2, 2)],
        [_document("b", 9999.0, 1), _document("a", -9999.0, 2)],
    )
    second = (
        [_document("a", 10_000.0, 1), _document("b", -1.0, 2)],
        [_document("b", 0.0, 1), _document("a", 1.0, 2)],
    )

    assert reciprocal_rank_fusion(first, rrf_k=60, top_k=2) == reciprocal_rank_fusion(
        second,
        rrf_k=60,
        top_k=2,
    )


def test_rrf_breaks_equal_scores_by_document_id() -> None:
    results = reciprocal_rank_fusion(
        ([_document("z", 1.0, 1)], [_document("a", 1.0, 1)]),
        rrf_k=60,
        top_k=2,
    )

    assert [result.doc_id for result in results] == ["a", "z"]


def test_hybrid_retriever_uses_configured_candidate_depths() -> None:
    bm25 = StaticRetriever(
        "bm25",
        [_document("shared", 1.0, 1), _document("sparse", 0.5, 2)],
    )
    dense = StaticRetriever(
        "dense",
        [_document("dense", 1.0, 1), _document("shared", 0.5, 2)],
    )
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        config=_config(),
    )

    results = retriever.retrieve("query", top_k=2)

    assert [result.doc_id for result in results] == ["shared", "dense"]
    assert bm25.calls == [("query", 3)]
    assert dense.calls == [("query", 2)]


def test_hybrid_retriever_runs_sparse_and_dense_retrieval_concurrently() -> None:
    barrier = Barrier(2)
    bm25 = BarrierRetriever("bm25", [_document("sparse", 1.0, 1)], barrier)
    dense = BarrierRetriever("dense", [_document("dense", 1.0, 1)], barrier)
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        config=_config(),
    )

    results = retriever.retrieve("query", top_k=2)

    assert {result.doc_id for result in results} == {"sparse", "dense"}


@pytest.mark.parametrize(
    "rankings",
    [
        ([], [_document("a", 1.0, 1)]),
        ([_document("a", 1.0, 2)], [_document("b", 1.0, 1)]),
        (
            [_document("a", 1.0, 1), _document("a", 0.5, 2)],
            [_document("b", 1.0, 1)],
        ),
    ],
)
def test_rrf_rejects_invalid_rankings(
    rankings: tuple[list[ScoredDocument], list[ScoredDocument]],
) -> None:
    with pytest.raises(HybridRetrievalError):
        reciprocal_rank_fusion(rankings, rrf_k=60, top_k=2)


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("query", 0), ("query", 3)])
def test_hybrid_retriever_rejects_invalid_inputs(query: str, top_k: int) -> None:
    retriever = HybridRetriever(
        bm25_retriever=StaticRetriever("bm25", [_document("a", 1.0, 1)]),
        dense_retriever=StaticRetriever("dense", [_document("b", 1.0, 1)]),
        config=_config(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(query, top_k)


def test_config_requires_candidate_depth_to_cover_top_k() -> None:
    with pytest.raises(ValueError, match="bm25_candidate_k"):
        HybridConfig(
            name="invalid",
            bm25_candidate_k=9,
            dense_candidate_k=10,
            rrf_k=60,
            top_k=10,
        )


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "hybrid.yaml"
    path.write_text(
        "name: invalid\nstrategy: hybrid_rrf\nbm25_candidate_k: 50\n"
        "dense_candidate_k: 50\nrrf_k: 60\ntop_k: 10\nweights: true\n",
        encoding="utf-8",
    )

    with pytest.raises(HybridRetrievalError, match="Unable to load hybrid config"):
        load_hybrid_config(path)
