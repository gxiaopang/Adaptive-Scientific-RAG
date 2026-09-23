"""Standardized retrieval metric calculation backed by ranx."""

import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Final, Protocol, cast


class _RanxEvaluate(Protocol):
    def __call__(
        self,
        qrels: dict[str, dict[str, int]],
        run: dict[str, dict[str, float]],
        metrics: list[str],
    ) -> object: ...

type RetrievalRun = Mapping[str, Mapping[str, float]]
type RetrievalQrels = Mapping[str, Mapping[str, int]]

DEFAULT_METRICS: Final[tuple[str, ...]] = (
    "recall@5",
    "recall@10",
    "precision@5",
    "precision@10",
    "mrr@10",
    "ndcg@10",
)


class RetrievalEvaluationError(ValueError):
    """Raised when qrels or a retrieval run cannot be compared fairly."""


def evaluate_retrieval(
    qrels: RetrievalQrels,
    run: RetrievalRun,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Evaluate one complete retrieval run with a stable metric vocabulary."""

    _validate_inputs(qrels, run, metrics)
    mutable_qrels = {
        query_id: dict(relevant_documents)
        for query_id, relevant_documents in qrels.items()
    }
    mutable_run = {query_id: dict(scores) for query_id, scores in run.items()}
    result = _load_ranx_evaluate()(mutable_qrels, mutable_run, list(metrics))
    if not isinstance(result, dict):
        raise RetrievalEvaluationError("Expected multiple metric results from ranx")
    return cast(dict[str, float], result)


def _validate_inputs(
    qrels: RetrievalQrels,
    run: RetrievalRun,
    metrics: Sequence[str],
) -> None:
    if not qrels:
        raise RetrievalEvaluationError("qrels cannot be empty")
    if not metrics:
        raise RetrievalEvaluationError("at least one metric is required")

    qrel_query_ids = set(qrels)
    run_query_ids = set(run)
    if qrel_query_ids != run_query_ids:
        missing = sorted(qrel_query_ids - run_query_ids)
        unexpected = sorted(run_query_ids - qrel_query_ids)
        raise RetrievalEvaluationError(
            f"run query ids must exactly match qrels; missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    for query_id, relevant_documents in qrels.items():
        if not relevant_documents:
            raise RetrievalEvaluationError(f"qrels query {query_id!r} has no relevant documents")
        if any(score <= 0 for score in relevant_documents.values()):
            raise RetrievalEvaluationError(
                f"qrels query {query_id!r} contains a non-positive relevance score"
            )

    for query_id, scores in run.items():
        if not scores:
            raise RetrievalEvaluationError(f"run query {query_id!r} has no retrieved documents")
        if any(not math.isfinite(score) for score in scores.values()):
            raise RetrievalEvaluationError(f"run query {query_id!r} contains a non-finite score")


def _load_ranx_evaluate() -> _RanxEvaluate:
    """Load the untyped third-party API behind a small, checked local boundary."""

    # ranx imports ir_datasets even for in-memory evaluation. ir_datasets writes
    # to its cache directory during import, so provide a writable fallback for
    # read-only containers while preserving an explicit user configuration.
    os.environ.setdefault(
        "IR_DATASETS_HOME",
        str(Path(tempfile.gettempdir()) / "adaptive-rag-ir-datasets"),
    )
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "adaptive-rag-matplotlib"),
    )
    return cast(_RanxEvaluate, import_module("ranx").evaluate)
