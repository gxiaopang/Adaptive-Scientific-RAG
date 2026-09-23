import math

import pytest

from adaptive_rag.embeddings.openai_compatible import (
    EmbeddingError,
    OpenAICompatibleEmbeddingProvider,
)


class FakeTransport:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def post(self, url: str, *, api_key: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((url, api_key, payload))
        return self.responses.pop(0)

    def close(self) -> None:
        return None


def _response(count: int, dimension: int = 3) -> dict[str, object]:
    return {
        "data": [
            {"index": index, "embedding": [float(index)] * dimension}
            for index in range(count)
        ]
    }


def _provider(transport: FakeTransport) -> OpenAICompatibleEmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider(
        provider="test",
        api_key="secret",
        base_url="https://provider.test/v1/",
        model="embedding-model",
        dimension=3,
        batch_size=2,
        timeout_seconds=1,
        max_retries=1,
        transport=transport,  # type: ignore[arg-type]
    )


def test_batches_documents_and_sends_explicit_dimension() -> None:
    transport = FakeTransport([_response(2), _response(1)])

    vectors = _provider(transport).encode_documents(["one", "two", "three"])

    assert len(vectors) == 3
    assert [call[2]["input"] for call in transport.calls] == [["one", "two"], ["three"]]
    assert transport.calls[0][0] == "https://provider.test/v1/embeddings"
    assert transport.calls[0][2]["dimensions"] == 3


@pytest.mark.parametrize(
    "response,match",
    [
        ({"data": []}, "vectors"),
        ({"data": [{"index": 1, "embedding": [0.0, 0.0, 0.0]}]}, "indexes"),
        ({"data": [{"index": 0, "embedding": [0.0]}]}, "dimension"),
        ({"data": [{"index": 0, "embedding": [0.0, math.nan, 0.0]}]}, "non-finite"),
    ],
)
def test_rejects_invalid_responses(response: dict[str, object], match: str) -> None:
    with pytest.raises(EmbeddingError, match=match):
        _provider(FakeTransport([response])).encode_query("query")


def test_empty_documents_do_not_call_provider() -> None:
    transport = FakeTransport([])
    assert _provider(transport).encode_documents([]) == []
    assert transport.calls == []
