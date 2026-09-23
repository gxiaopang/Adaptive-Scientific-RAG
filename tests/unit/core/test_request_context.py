from adaptive_rag.core.context import (
    bind_request_context,
    get_request_context,
    reset_request_context,
)


def test_request_context_is_nested_and_reset_with_tokens() -> None:
    outer = bind_request_context(request_id="outer-request", trace_id="outer-trace")
    try:
        inner = bind_request_context(request_id="inner-request", trace_id="inner-trace")
        assert get_request_context().request_id == "inner-request"
        reset_request_context(inner)
        assert get_request_context().request_id == "outer-request"
        assert get_request_context().trace_id == "outer-trace"
    finally:
        reset_request_context(outer)

    assert get_request_context().request_id is None
    assert get_request_context().trace_id is None
