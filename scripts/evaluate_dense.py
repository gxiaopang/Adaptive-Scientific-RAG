"""Evaluate the pgvector dense baseline on fixed SciFact validation queries."""

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
from adaptive_rag.domain.protocols import ProviderInvocationError
from adaptive_rag.evaluation.metrics import RetrievalEvaluationError
from adaptive_rag.evaluation.runner import evaluate_retriever
from adaptive_rag.retrieval.dense import (
    DenseIndexError,
    DenseRetriever,
    load_dense_config,
    verify_dense_index,
)
from adaptive_rag.runtime import build_embedding_provider, build_postgres_dependencies
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore, PgVectorStoreError

DEFAULT_CONFIG = Path("configs/dense.yaml")
DEFAULT_OUTPUT = Path(".artifacts/evaluations/dense_validation.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    postgres = None
    embedder = None
    try:
        dataset = load_scifact(args.data_dir or settings.scifact_data_dir)
        report = validate_scifact(dataset)
        split = create_development_split(
            dataset.qrels["train"],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        config = load_dense_config(args.config)
        postgres = build_postgres_dependencies(settings)
        embedder = build_embedding_provider(settings, config)
        store = PgVectorDenseStore(
            postgres.sessions, index_name=config.index_name, dimension=config.dimension
        )
        snapshot = verify_dense_index(
            corpus=dataset.corpus,
            corpus_checksum=report.checksums["corpus.jsonl"],
            config=config,
            embedder=embedder,
            store=store,
        )
        result = evaluate_retriever(
            DenseRetriever(embedder=embedder, store=store, config=config),
            queries=dataset.queries,
            qrels=dataset.qrels["train"],
            query_ids=split.validation_query_ids,
            top_k=config.top_k,
        )
    except (
        DataFormatError,
        DatasetValidationError,
        DenseIndexError,
        PgVectorStoreError,
        ProviderInvocationError,
        RetrievalEvaluationError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if embedder is not None:
            embedder.close()
        if postgres is not None:
            postgres.engine.dispose()

    payload = {
        "dataset": "scifact",
        "split": "validation",
        "seed": split.seed,
        "validation_fraction": split.validation_fraction,
        "index_metadata": asdict(snapshot),
        "result": asdict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
