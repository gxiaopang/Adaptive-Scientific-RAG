import pytest

from adaptive_rag.domain.models import EvidenceDocument
from adaptive_rag.graph.context import TRUNCATION_MARKER, build_evidence_context


def _evidence() -> tuple[EvidenceDocument, ...]:
    return (
        EvidenceDocument("d1", "First title", "First body", 0.9, 1),
        EvidenceDocument("d2", "", "Second body", 0.8, 2),
    )


def test_builds_ranked_document_labelled_context() -> None:
    context = build_evidence_context(_evidence(), maximum_characters=1_000)

    assert context == (
        "[document:d1 rank:1]\nFirst title\nFirst body\n\n"
        "[document:d2 rank:2]\nSecond body"
    )


def test_truncates_context_at_exact_character_budget() -> None:
    context = build_evidence_context(_evidence(), maximum_characters=50)

    assert len(context) <= 50
    assert context.endswith(TRUNCATION_MARKER)


def test_rejects_empty_evidence() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        build_evidence_context((), maximum_characters=1_000)


def test_rejects_budget_too_small_for_truncation_marker() -> None:
    with pytest.raises(ValueError, match="too small"):
        build_evidence_context(
            _evidence(),
            maximum_characters=len(TRUNCATION_MARKER),
        )
