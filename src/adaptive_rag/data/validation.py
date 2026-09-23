"""Cross-file integrity checks and deterministic development splits for SciFact."""

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from adaptive_rag.data.scifact import Qrels, SciFactDataset


class DatasetValidationError(ValueError):
    """Raised with all detected cross-file integrity violations."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("SciFact validation failed: " + "; ".join(issues))


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    corpus_count: int
    query_count: int
    split_query_counts: dict[str, int]
    split_qrels_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    statistics: DatasetStatistics
    checksums: dict[str, str]


@dataclass(frozen=True, slots=True)
class DevelopmentSplit:
    seed: int
    validation_fraction: float
    calibration_query_ids: tuple[str, ...]
    validation_query_ids: tuple[str, ...]


def validate_scifact(dataset: SciFactDataset) -> ValidationReport:
    """Validate query/document references, split isolation, and source checksums."""

    issues: list[str] = []
    corpus_ids = set(dataset.corpus)
    query_ids = set(dataset.queries)
    seen_split_queries: set[str] = set()
    split_query_counts: dict[str, int] = {}
    split_qrels_counts: dict[str, int] = {}

    for split_name, split_qrels in dataset.qrels.items():
        split_query_ids = set(split_qrels)
        overlap = seen_split_queries & split_query_ids
        if overlap:
            issues.append(
                f"qrels split {split_name!r} overlaps earlier splits: {_sample_ids(overlap)}"
            )
        seen_split_queries.update(split_query_ids)

        missing_queries = split_query_ids - query_ids
        if missing_queries:
            issues.append(
                f"qrels split {split_name!r} references unknown queries: "
                f"{_sample_ids(missing_queries)}"
            )

        referenced_documents = {
            doc_id for relevant_documents in split_qrels.values() for doc_id in relevant_documents
        }
        missing_documents = referenced_documents - corpus_ids
        if missing_documents:
            issues.append(
                f"qrels split {split_name!r} references unknown documents: "
                f"{_sample_ids(missing_documents)}"
            )

        split_query_counts[split_name] = len(split_query_ids)
        split_qrels_counts[split_name] = sum(map(len, split_qrels.values()))

    queries_without_qrels = query_ids - seen_split_queries
    if queries_without_qrels:
        issues.append(f"queries have no qrels: {_sample_ids(queries_without_qrels)}")

    if issues:
        raise DatasetValidationError(issues)

    return ValidationReport(
        statistics=DatasetStatistics(
            corpus_count=len(corpus_ids),
            query_count=len(query_ids),
            split_query_counts=split_query_counts,
            split_qrels_counts=split_qrels_counts,
        ),
        checksums=_source_checksums(dataset),
    )


def create_development_split(
    train_qrels: Qrels,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DevelopmentSplit:
    """Split train queries deterministically, stratified by relevant-document count."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(train_qrels) < 2:
        raise ValueError("At least two train queries are required")

    strata: dict[int, list[str]] = defaultdict(list)
    for query_id, relevant_documents in train_qrels.items():
        if not relevant_documents:
            raise ValueError(f"Train query {query_id!r} has no relevant documents")
        strata[len(relevant_documents)].append(query_id)

    target_validation_count = round(len(train_qrels) * validation_fraction)
    target_validation_count = min(max(target_validation_count, 1), len(train_qrels) - 1)
    allocations = _allocate_validation_counts(
        strata,
        validation_fraction=validation_fraction,
        target_count=target_validation_count,
    )

    validation_ids: set[str] = set()
    for relevance_count, query_ids in strata.items():
        ordered_ids = sorted(query_ids, key=lambda query_id: _split_key(seed, query_id))
        validation_ids.update(ordered_ids[: allocations[relevance_count]])

    all_query_ids = set(train_qrels)
    calibration_ids = all_query_ids - validation_ids
    return DevelopmentSplit(
        seed=seed,
        validation_fraction=validation_fraction,
        calibration_query_ids=tuple(sorted(calibration_ids)),
        validation_query_ids=tuple(sorted(validation_ids)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DatasetValidationError([f"Unable to checksum {path}: {error}"]) from error
    return digest.hexdigest()


def _source_checksums(dataset: SciFactDataset) -> dict[str, str]:
    paths = {
        "corpus.jsonl": dataset.data_dir / "corpus.jsonl",
        "queries.jsonl": dataset.data_dir / "queries.jsonl",
        **{
            f"qrels/{split_name}.tsv": dataset.data_dir / "qrels" / f"{split_name}.tsv"
            for split_name in dataset.qrels
        },
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _allocate_validation_counts(
    strata: dict[int, list[str]],
    *,
    validation_fraction: float,
    target_count: int,
) -> dict[int, int]:
    exact_counts = {
        relevance_count: len(query_ids) * validation_fraction
        for relevance_count, query_ids in strata.items()
    }
    allocations = {
        relevance_count: math.floor(exact_count)
        for relevance_count, exact_count in exact_counts.items()
    }
    remaining = target_count - sum(allocations.values())
    largest_remainders = sorted(
        strata,
        key=lambda relevance_count: (
            -(exact_counts[relevance_count] - allocations[relevance_count]),
            -len(strata[relevance_count]),
            relevance_count,
        ),
    )
    for relevance_count in largest_remainders[:remaining]:
        allocations[relevance_count] += 1
    return allocations


def _split_key(seed: int, query_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{query_id}".encode()).digest()


def _sample_ids(ids: set[str], limit: int = 5) -> str:
    ordered_ids = sorted(ids)
    suffix = " ..." if len(ordered_ids) > limit else ""
    return ", ".join(ordered_ids[:limit]) + suffix

