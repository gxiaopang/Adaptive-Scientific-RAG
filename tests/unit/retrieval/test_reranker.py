import math
from pathlib import Path

import pytest

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.retrieval.reranker import (
    RerankerConfig,
    RerankerError,
    RerankingRetriever,
    load_reranker_config,
)


class FakePairwiseReranker:
    provider = "test-provider"
    model_name = "test/reranker"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return self.scores


class FakeCandidateRetriever:
    strategy = "hybrid_rrf"

    def __init__(self, results: list[ScoredDocument]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        return self.results[:top_k]


def _config() -> RerankerConfig:
    return RerankerConfig(
        name="test-reranker",
        provider="test-provider",
        model_name="test/reranker",
        candidate_k=3,
        top_k=2,
    )


def _corpus() -> dict[str, CorpusDocument]:
    return {
        "d1": CorpusDocument("d1", "Title one", "Body one", {}),
        "d2": CorpusDocument("d2", "Title two", "Body two", {}),
        "d3": CorpusDocument("d3", "", "Body three", {}),
    }


def _candidates() -> list[ScoredDocument]:
    return [
        ScoredDocument("d1", 0.3, 1),
        ScoredDocument("d2", 0.2, 2),
        ScoredDocument("d3", 0.1, 3),
    ]


def test_reranks_canonical_text_with_deterministic_ties() -> None:
    pairwise = FakePairwiseReranker([0.5, 0.9, 0.5])
    retriever = RerankingRetriever(
        candidate_retriever=FakeCandidateRetriever(_candidates()),
        reranker=pairwise,
        corpus=_corpus(),
        config=_config(),
    )

    results = retriever.retrieve("scientific query", 3)

    assert [result.doc_id for result in results] == ["d2", "d1", "d3"]
    assert pairwise.calls[0][1] == [
        "Title one\n\nBody one",
        "Title two\n\nBody two",
        "Body three",
    ]


@pytest.mark.parametrize("scores", [[1.0], [1.0, math.nan, 0.0]])
def test_invalid_scores_fail_closed(scores: list[float]) -> None:
    retriever = RerankingRetriever(
        candidate_retriever=FakeCandidateRetriever(_candidates()),
        reranker=FakePairwiseReranker(scores),
        corpus=_corpus(),
        config=_config(),
    )
    with pytest.raises(RerankerError):
        retriever.retrieve("query", 2)


def test_provider_identity_must_match() -> None:
    reranker = FakePairwiseReranker([1.0])
    reranker.provider = "other"
    with pytest.raises(RerankerError, match="provider"):
        RerankingRetriever(
            candidate_retriever=FakeCandidateRetriever(_candidates()),
            reranker=reranker,
            corpus=_corpus(),
            config=_config(),
        )


def test_config_requires_candidate_depth_to_cover_output() -> None:
    with pytest.raises(ValueError, match="candidate_k"):
        RerankerConfig(
            name="invalid",
            provider="test-provider",
            model_name="test/reranker",
            candidate_k=9,
            top_k=10,
        )


def test_load_config_rejects_local_model_fields(tmp_path: Path) -> None:
    path = tmp_path / "reranker.yaml"
    path.write_text(
        "name: invalid\nstrategy: hybrid_rrf_reranker\nprovider: test\n"
        "model_name: model\ncandidate_k: 50\ntop_k: 10\ndevice: cpu\n"
    )
    with pytest.raises(RerankerError, match="Unable to load reranker config"):
        load_reranker_config(path)
