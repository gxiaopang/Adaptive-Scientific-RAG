"""Dataset loading and validation contracts."""

from adaptive_rag.data.scifact import (
    CorpusDocument,
    DataFormatError,
    Qrels,
    Query,
    SciFactDataset,
    load_scifact,
)

__all__ = [
    "CorpusDocument",
    "DataFormatError",
    "Qrels",
    "Query",
    "SciFactDataset",
    "load_scifact",
]
