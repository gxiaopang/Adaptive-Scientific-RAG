import math

import pytest

from adaptive_rag.retrieval.online_reranker import OnlineRerankerError, QwenTextReranker


class FakeTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.payload: dict[str, object] | None = None

    def post(self, url: str, *, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return self.response

    def close(self) -> None:
        return None


def _reranker(response: dict[str, object]) -> tuple[QwenTextReranker, FakeTransport]:
    transport = FakeTransport(response)
    reranker = QwenTextReranker(
        provider="test",
        api_key="secret",
        url="https://provider.test/rerank",
        model="rerank-model",
        instruction="Retrieve evidence.",
        timeout_seconds=1,
        max_retries=1,
        transport=transport,  # type: ignore[arg-type]
    )
    return reranker, transport


def test_maps_provider_indexes_back_to_input_order() -> None:
    reranker, transport = _reranker(
        {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        }
    )

    assert reranker.score("query", ["first", "second"]) == [0.2, 0.9]
    assert transport.payload is not None
    assert transport.payload["parameters"] == {"top_n": 2, "instruct": "Retrieve evidence."}


@pytest.mark.parametrize(
    "results,match",
    [
        ([{"index": 0, "relevance_score": 0.9}], "results for"),
        (
            [
                {"index": 0, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            "duplicate",
        ),
        (
            [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ],
            "index",
        ),
        (
            [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ],
            "sorted",
        ),
        (
            [
                {"index": 0, "relevance_score": math.nan},
                {"index": 1, "relevance_score": 0.1},
            ],
            "non-finite",
        ),
    ],
)
def test_rejects_invalid_provider_results(results: list[dict[str, object]], match: str) -> None:
    reranker, _ = _reranker({"output": {"results": results}})
    with pytest.raises(OnlineRerankerError, match=match):
        reranker.score("query", ["first", "second"])
