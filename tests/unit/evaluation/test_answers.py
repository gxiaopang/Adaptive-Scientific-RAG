import os
from typing import TypeVar

import pytest
from pydantic import BaseModel

from adaptive_rag.evaluation.answers import (
    AnswerJudge,
    finite_score,
    make_ragas_embeddings,
    precision_recall,
)

Output = TypeVar("Output", bound=BaseModel)


def test_precision_recall_uses_fixed_k_and_label_denominator() -> None:
    assert precision_recall(["a", "b"], {"a", "c"}, 5) == {
        "precision@5": 0.2, "recall@5": 0.5,
    }
    with pytest.raises(ValueError):
        precision_recall(["a", "a"], {"a"}, 5)
    assert precision_recall([], {"a"}, 10) == {"precision@10": 0, "recall@10": 0}


def test_undefined_metric_is_not_zero_or_one() -> None:
    assert finite_score(float("nan")) is None
    assert finite_score(0.0) == 0.0
    with pytest.raises(ValueError):
        finite_score("0.8")


class FakeEmbedding:
    provider = "fake"
    model_name = "fake"
    dimension = 2

    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeModel:
    def __init__(self, empty_claims: bool = False) -> None:
        self.empty_claims = empty_claims
        self.calls: list[str] = []

    def complete_structured(self, messages: object, response_model: type[Output]) -> Output:
        name = response_model.__name__
        self.calls.append(name)
        if name == "AnswerRelevanceOutput":
            value = {"question": "Does A cause B?", "noncommittal": 0}
        elif name == "StatementGeneratorOutput":
            value = {"statements": [] if self.empty_claims else ["A causes B.", "C causes D."]}
        else:
            value = {"statements": [
                {"statement": "A causes B.", "reason": "supported", "verdict": 1},
                {"statement": "C causes D.", "reason": "not supported", "verdict": 0},
            ]}
        return response_model.model_validate(value)

    def complete(self, messages: object) -> str:
        raise AssertionError("only structured judging is expected")


@pytest.mark.parametrize("empty_claims", [False, True])
def test_real_ragas_metrics_with_offline_fakes(empty_claims: bool) -> None:
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    pytest.importorskip("ragas")
    model = FakeModel(empty_claims)
    judge = AnswerJudge(model, make_ragas_embeddings(FakeEmbedding()))
    result = judge.evaluate("Does A cause B?", "A causes B. C causes D.", "A causes B.")
    assert result["answer_relevancy"] == pytest.approx(1.0)
    assert result["faithfulness"] == (None if empty_claims else 0.5)
    assert model.calls.count("AnswerRelevanceOutput") == 3
    assert len(result["ragas_trace"]) == (4 if empty_claims else 5)
