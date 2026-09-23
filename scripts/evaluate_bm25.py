"""Evaluate the persisted BM25 baseline on the fixed SciFact validation queries."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from adaptive_rag.core.config import get_settings
from adaptive_rag.data.scifact import DataFormatError, load_scifact
from adaptive_rag.data.validation import (
    DatasetValidationError,
    create_development_split,
    validate_scifact,
)
from adaptive_rag.evaluation.metrics import RetrievalEvaluationError
from adaptive_rag.evaluation.runner import evaluate_retriever
from adaptive_rag.retrieval.bm25 import BM25IndexError, BM25Retriever

DEFAULT_INDEX_DIR = Path(".artifacts/indexes/bm25/scifact-v1")
DEFAULT_OUTPUT = Path(".artifacts/evaluations/bm25_validation.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or get_settings().scifact_data_dir
    try:
        dataset = load_scifact(data_dir)
        validation_report = validate_scifact(dataset)
        development_split = create_development_split(
            dataset.qrels["train"],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        retriever = BM25Retriever.load(args.index_dir, mmap=True)
        expected_checksum = validation_report.checksums["corpus.jsonl"]
        if retriever.manifest.corpus_checksum != expected_checksum:
            raise BM25IndexError("BM25 index corpus checksum does not match the current dataset")
        result = evaluate_retriever(
            retriever,
            queries=dataset.queries,
            qrels=dataset.qrels["train"],
            query_ids=development_split.validation_query_ids,
            top_k=retriever.manifest.config.top_k,
        )
    except (
        DataFormatError,
        DatasetValidationError,
        BM25IndexError,
        RetrievalEvaluationError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = {
        "dataset": "scifact",
        "split": "validation",
        "seed": development_split.seed,
        "validation_fraction": development_split.validation_fraction,
        "index_manifest": retriever.manifest.model_dump(mode="json"),
        "result": asdict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "strategy": result.strategy,
                "query_count": result.query_count,
                "top_k": result.top_k,
                "metrics": result.metrics,
                "latency_ms": asdict(result.latency),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

