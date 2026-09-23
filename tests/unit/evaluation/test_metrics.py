import pytest

from adaptive_rag.evaluation.metrics import (
    DEFAULT_METRICS,
    RetrievalEvaluationError,
    evaluate_retrieval,
)


def test_evaluate_retrieval_matches_hand_calculated_metrics() -> None:
    qrels = {"q1": {"d1": 1, "d2": 1}, "q2": {"d4": 1}}
    run = {
        "q1": {"d1": 3.0, "d3": 2.0, "d2": 1.0},
        "q2": {"d5": 2.0, "d4": 1.0},
    }

    result = evaluate_retrieval(qrels, run)

    assert tuple(result) == DEFAULT_METRICS
    assert result["recall@5"] == pytest.approx(1.0)
    assert result["recall@10"] == pytest.approx(1.0)
    assert result["precision@5"] == pytest.approx(0.3)
    assert result["precision@10"] == pytest.approx(0.15)
    assert result["mrr@10"] == pytest.approx(0.75)
    assert result["ndcg@10"] == pytest.approx(0.77532527)


def test_evaluate_retrieval_requires_complete_query_coverage() -> None:
    with pytest.raises(RetrievalEvaluationError, match="must exactly match"):
        evaluate_retrieval({"q1": {"d1": 1}, "q2": {"d2": 1}}, {"q1": {"d1": 1.0}})


def test_evaluate_retrieval_rejects_non_finite_scores() -> None:
    with pytest.raises(RetrievalEvaluationError, match="non-finite"):
        evaluate_retrieval({"q1": {"d1": 1}}, {"q1": {"d1": float("nan")}})

