"""Request-scoped correlation context propagated through ContextVar."""

from contextvars import ContextVar, Token
from dataclasses import dataclass

_REQUEST_ID: ContextVar[str | None] = ContextVar("adaptive_rag_request_id", default=None)
_TRACE_ID: ContextVar[str | None] = ContextVar("adaptive_rag_trace_id", default=None)


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str | None
    trace_id: str | None


@dataclass(frozen=True, slots=True)
class RequestContextTokens:
    request_id: Token[str | None]
    trace_id: Token[str | None]


def bind_request_context(*, request_id: str, trace_id: str) -> RequestContextTokens:
    """Bind identifiers for the current async task and return reset tokens."""

    return RequestContextTokens(
        request_id=_REQUEST_ID.set(request_id),
        trace_id=_TRACE_ID.set(trace_id),
    )


def reset_request_context(tokens: RequestContextTokens) -> None:
    """Restore the context that existed before one request was bound."""

    _TRACE_ID.reset(tokens.trace_id)
    _REQUEST_ID.reset(tokens.request_id)


def get_request_context() -> RequestContext:
    """Return correlation identifiers visible in the current execution context."""

    return RequestContext(request_id=_REQUEST_ID.get(), trace_id=_TRACE_ID.get())
