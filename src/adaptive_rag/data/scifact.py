"""Typed loader for the BEIR representation of the SciFact dataset."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

type Qrels = dict[str, dict[str, int]]


class DataFormatError(ValueError):
    """Raised when a SciFact source file violates its structural contract."""


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One document in the BEIR corpus."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, object]

    @property
    def retrieval_text(self) -> str:
        """Return the single canonical text representation used by later retrievers."""

        title = self.title.strip()
        text = self.text.strip()
        return f"{title}\n\n{text}" if title else text


@dataclass(frozen=True, slots=True)
class Query:
    """One BEIR retrieval query."""

    query_id: str
    text: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class SciFactDataset:
    """The complete corpus, queries, and qrels loaded from one data directory."""

    data_dir: Path
    corpus: dict[str, CorpusDocument]
    queries: dict[str, Query]
    qrels: dict[str, Qrels]


class _CorpusRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    doc_id: str = Field(alias="_id", min_length=1)
    title: str = ""
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class _QueryRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query_id: str = Field(alias="_id", min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)


def load_scifact(data_dir: Path, splits: tuple[str, ...] = ("train", "test")) -> SciFactDataset:
    """Load a SciFact directory without changing or preprocessing its source files."""

    resolved_data_dir = data_dir.resolve()
    if not resolved_data_dir.is_dir():
        raise DataFormatError(f"SciFact data directory does not exist: {resolved_data_dir}")
    if not splits:
        raise ValueError("At least one qrels split is required")
    if len(set(splits)) != len(splits):
        raise ValueError("Qrels split names must be unique")

    corpus = _load_corpus(resolved_data_dir / "corpus.jsonl")
    queries = _load_queries(resolved_data_dir / "queries.jsonl")
    qrels = {
        split: _load_qrels(resolved_data_dir / "qrels" / f"{split}.tsv") for split in splits
    }
    return SciFactDataset(
        data_dir=resolved_data_dir,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )


def _load_corpus(path: Path) -> dict[str, CorpusDocument]:
    rows: dict[str, CorpusDocument] = {}
    for line_number, raw_row in _read_jsonl(path):
        try:
            parsed = _CorpusRow.model_validate(raw_row)
        except ValidationError as error:
            raise DataFormatError(_validation_message(path, line_number, error)) from error
        if parsed.doc_id in rows:
            raise DataFormatError(f"{path}:{line_number}: duplicate corpus id {parsed.doc_id!r}")
        rows[parsed.doc_id] = CorpusDocument(
            doc_id=parsed.doc_id,
            title=parsed.title,
            text=parsed.text,
            metadata=parsed.metadata,
        )
    if not rows:
        raise DataFormatError(f"{path}: corpus is empty")
    return rows


def _load_queries(path: Path) -> dict[str, Query]:
    rows: dict[str, Query] = {}
    for line_number, raw_row in _read_jsonl(path):
        try:
            parsed = _QueryRow.model_validate(raw_row)
        except ValidationError as error:
            raise DataFormatError(_validation_message(path, line_number, error)) from error
        if parsed.query_id in rows:
            raise DataFormatError(f"{path}:{line_number}: duplicate query id {parsed.query_id!r}")
        rows[parsed.query_id] = Query(
            query_id=parsed.query_id,
            text=parsed.text,
            metadata=parsed.metadata,
        )
    if not rows:
        raise DataFormatError(f"{path}: queries are empty")
    return rows


def _read_jsonl(path: Path) -> list[tuple[int, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise DataFormatError(f"Unable to read {path}: {error}") from error

    rows: list[tuple[int, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DataFormatError(f"{path}:{line_number}: blank JSONL row")
        try:
            rows.append((line_number, cast(object, json.loads(line))))
        except json.JSONDecodeError as error:
            raise DataFormatError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
    return rows


def _load_qrels(path: Path) -> Qrels:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as error:
        raise DataFormatError(f"Unable to read {path}: {error}") from error

    qrels: Qrels = {}
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_header = ["query-id", "corpus-id", "score"]
        if reader.fieldnames != expected_header:
            raise DataFormatError(
                f"{path}: expected TSV header {expected_header}, got {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            query_id = row.get("query-id", "").strip()
            corpus_id = row.get("corpus-id", "").strip()
            raw_score = row.get("score", "").strip()
            if not query_id or not corpus_id:
                raise DataFormatError(f"{path}:{line_number}: qrels ids cannot be empty")
            try:
                score = int(raw_score)
            except ValueError as error:
                raise DataFormatError(
                    f"{path}:{line_number}: score must be an integer, got {raw_score!r}"
                ) from error
            if score <= 0:
                raise DataFormatError(f"{path}:{line_number}: score must be positive")

            query_rels = qrels.setdefault(query_id, {})
            if corpus_id in query_rels:
                raise DataFormatError(
                    f"{path}:{line_number}: duplicate qrels pair ({query_id!r}, {corpus_id!r})"
                )
            query_rels[corpus_id] = score

    if not qrels:
        raise DataFormatError(f"{path}: qrels are empty")
    return qrels


def _validation_message(path: Path, line_number: int, error: ValidationError) -> str:
    first_error = error.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first_error["loc"])
    return f"{path}:{line_number}: invalid {location}: {first_error['msg']}"

