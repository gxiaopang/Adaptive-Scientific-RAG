"""Deterministic evidence packing for bounded generator input."""

from collections.abc import Sequence

from adaptive_rag.domain.models import EvidenceDocument

TRUNCATION_MARKER = "\n...[context truncated]"


def build_evidence_context(
    evidence: Sequence[EvidenceDocument],
    *,
    maximum_characters: int,
) -> str:
    """Render ranked evidence into a bounded, document-labelled context string."""

    if not evidence:
        raise ValueError("evidence cannot be empty")
    if maximum_characters < len(TRUNCATION_MARKER) + 1:
        raise ValueError("maximum_characters is too small for a bounded context")

    blocks = [_render_document(document) for document in evidence]
    context = "\n\n".join(blocks)
    if len(context) <= maximum_characters:
        return context
    content_budget = maximum_characters - len(TRUNCATION_MARKER)
    return context[:content_budget].rstrip() + TRUNCATION_MARKER


def _render_document(document: EvidenceDocument) -> str:
    title = document.title.strip()
    body = document.text.strip()
    header = f"[document:{document.doc_id} rank:{document.rank}]"
    content = f"{title}\n{body}" if title else body
    return f"{header}\n{content}"
