"""Provider-independent domain contracts."""

from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.domain.protocols import Retriever

__all__ = ["Retriever", "ScoredDocument"]

