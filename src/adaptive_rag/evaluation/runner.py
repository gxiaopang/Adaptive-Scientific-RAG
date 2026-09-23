"""Strategy-neutral offline retrieval runner with per-query latency measurement."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter_ns

from adaptive_rag.data.scifact import Qrels, Query
from adaptive_rag.domain.protocols import Retriever
from adaptive_rag.evaluation.metrics import evaluate_retrieval


@dataclass(frozen=True, slots=True)
class LatencySummary:
    mean_ms: float
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    strategy: str
    query_count: int
    top_k: int
    metrics: dict[str, float]
    latency: LatencySummary
    per_query_latency_ms: dict[str, float]
    run: dict[str, dict[str, float]]


def evaluate_retriever(
    retriever: Retriever,
    *,
    queries: Mapping[str, Query],
    qrels: Qrels,
    query_ids: Sequence[str],
    top_k: int,
) -> RetrievalEvaluationResult:
    """Retrieve every selected query exactly once and evaluate the complete run."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not query_ids:
        raise ValueError("query_ids cannot be empty")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("query_ids must be unique")

    selected_qrels: Qrels = {}
    run: dict[str, dict[str, float]] = {}
    latencies: dict[str, float] = {}
    for query_id in query_ids:
        if query_id not in queries:
            raise ValueError(f"Unknown query id: {query_id}")
        if query_id not in qrels:
            raise ValueError(f"Query id has no qrels: {query_id}")

        started_at = perf_counter_ns()
        results = retriever.retrieve(queries[query_id].text, top_k)
        elapsed_ms = (perf_counter_ns() - started_at) / 1_000_000
        _validate_results(query_id, results, top_k)

        selected_qrels[query_id] = qrels[query_id]
        run[query_id] = {result.doc_id: result.score for result in results}
        latencies[query_id] = elapsed_ms

    return RetrievalEvaluationResult(
        strategy=retriever.strategy,
        query_count=len(query_ids),
        top_k=top_k,
        metrics=evaluate_retrieval(selected_qrels, run),
        latency=_summarize_latency(tuple(latencies.values())),
        per_query_latency_ms=latencies,
        run=run,
    )


def _validate_results(query_id: str, results: Sequence[object], top_k: int) -> None:
    if not results:
        raise ValueError(f"Retriever returned no results for query {query_id}")
    if len(results) > top_k:
        raise ValueError(f"Retriever exceeded top_k for query {query_id}")

    doc_ids: set[str] = set()
    for expected_rank, result in enumerate(results, start=1):
        from adaptive_rag.domain.models import ScoredDocument

        if not isinstance(result, ScoredDocument):
            raise TypeError("Retriever results must be ScoredDocument instances")
        if result.rank != expected_rank:
            raise ValueError(f"Retriever returned non-consecutive ranks for query {query_id}")
        if result.doc_id in doc_ids:
            raise ValueError(f"Retriever returned duplicate documents for query {query_id}")
        if not math.isfinite(result.score):
            raise ValueError(f"Retriever returned a non-finite score for query {query_id}")
        doc_ids.add(result.doc_id)


def _summarize_latency(values: Sequence[float]) -> LatencySummary:
    ordered = sorted(values)
    return LatencySummary(
        mean_ms=sum(ordered) / len(ordered),
        median_ms=_percentile(ordered, 0.5),
        p95_ms=_percentile(ordered, 0.95),
    )


def _percentile(ordered_values: Sequence[float], quantile: float) -> float:
    position = (len(ordered_values) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    weight = position - lower_index
    return (
        ordered_values[lower_index] * (1.0 - weight) + ordered_values[upper_index] * weight
    )

