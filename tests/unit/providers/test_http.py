from collections.abc import Mapping

import httpx
import pytest

from adaptive_rag.providers.http import ProviderHttpError, RetryingJsonClient


class FakeClient:
    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: float,
    ) -> httpx.Response:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        return None


def _response(status: int, body: object = None) -> httpx.Response:
    return httpx.Response(status, json={} if body is None else body)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_rate_limit_and_server_errors(status: int) -> None:
    client = FakeClient([_response(status), _response(200, {"ok": True})])
    sleeps: list[float] = []
    transport = RetryingJsonClient(
        timeout_seconds=1,
        max_retries=1,
        client=client,
        sleep=sleeps.append,
    )

    assert transport.post("https://provider.test", api_key="secret", payload={}) == {"ok": True}
    assert client.calls == 2
    assert sleeps == [0.5]


def test_retries_timeout_and_sanitizes_final_error() -> None:
    client = FakeClient([httpx.ReadTimeout("secret input"), httpx.ReadTimeout("secret input")])
    transport = RetryingJsonClient(
        timeout_seconds=1,
        max_retries=1,
        client=client,
        sleep=lambda _: None,
    )

    with pytest.raises(ProviderHttpError, match="timed out") as caught:
        transport.post("https://provider.test", api_key="top-secret", payload={"text": "private"})
    assert "top-secret" not in str(caught.value)
    assert "private" not in str(caught.value)


def test_non_retryable_error_fails_immediately() -> None:
    client = FakeClient([_response(400), _response(200)])
    transport = RetryingJsonClient(timeout_seconds=1, max_retries=2, client=client)
    with pytest.raises(ProviderHttpError, match="400"):
        transport.post("https://provider.test", api_key="secret", payload={})
    assert client.calls == 1
