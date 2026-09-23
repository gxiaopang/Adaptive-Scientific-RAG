from adaptive_rag.data.scifact import Query
from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.evaluation.runner import evaluate_retriever


class FakeRetriever:
    strategy = "fake"

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        results = {
            "first": [ScoredDocument("d1", 2.0, 1), ScoredDocument("d3", 1.0, 2)],
            "second": [ScoredDocument("d2", 3.0, 1), ScoredDocument("d4", 1.0, 2)],
        }
        return results[query][:top_k]


def test_evaluate_retriever_builds_complete_run_and_latency() -> None:
    queries = {"q1": Query("q1", "first", {}), "q2": Query("q2", "second", {})}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}

    result = evaluate_retriever(
        FakeRetriever(),
        queries=queries,
        qrels=qrels,
        query_ids=("q1", "q2"),
        top_k=2,
    )

    assert result.strategy == "fake"
    assert result.query_count == 2
    assert result.run == {"q1": {"d1": 2.0, "d3": 1.0}, "q2": {"d2": 3.0, "d4": 1.0}}
    assert result.metrics["recall@5"] == 1.0
    assert result.latency.mean_ms >= 0.0
    assert set(result.per_query_latency_ms) == {"q1", "q2"}

