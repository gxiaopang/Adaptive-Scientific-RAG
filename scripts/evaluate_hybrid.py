"""Evaluate BM25 plus pgvector dense RRF on SciFact validation queries."""

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
from adaptive_rag.retrieval.bm25 import BM25IndexError, BM25Retriever
from adaptive_rag.retrieval.dense import (
    DenseIndexError,
    DenseRetriever,
    load_dense_config,
    verify_dense_index,
)
from adaptive_rag.retrieval.hybrid import HybridRetrievalError, HybridRetriever, load_hybrid_config
from adaptive_rag.runtime import build_embedding_provider, build_postgres_dependencies
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore, PgVectorStoreError

DEFAULT_CONFIG = Path("configs/hybrid.yaml")
DEFAULT_DENSE_CONFIG = Path("configs/dense.yaml")
DEFAULT_BM25_INDEX_DIR = Path(".artifacts/indexes/bm25/scifact-v1")
DEFAULT_OUTPUT = Path(".artifacts/evaluations/hybrid_rrf_validation.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dense-config", type=Path, default=DEFAULT_DENSE_CONFIG)
    parser.add_argument("--bm25-index-dir", type=Path, default=DEFAULT_BM25_INDEX_DIR)
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
        config = load_hybrid_config(args.config)
        dense_config = load_dense_config(args.dense_config)
        bm25 = BM25Retriever.load(args.bm25_index_dir, mmap=True)
        corpus_checksum = report.checksums["corpus.jsonl"]
        if bm25.manifest.corpus_checksum != corpus_checksum:
            raise HybridRetrievalError("BM25 index corpus checksum does not match the dataset")
        postgres = build_postgres_dependencies(settings)
        embedder = build_embedding_provider(settings, dense_config)
        store = PgVectorDenseStore(
            postgres.sessions,
            index_name=dense_config.index_name,
            dimension=dense_config.dimension,
        )
        snapshot = verify_dense_index(
            corpus=dataset.corpus,
            corpus_checksum=corpus_checksum,
            config=dense_config,
            embedder=embedder,
            store=store,
        )
        if bm25.manifest.doc_ids_checksum != snapshot.doc_ids_checksum:
            raise HybridRetrievalError("BM25 and dense indexes contain different documents")
        retriever = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=DenseRetriever(
                embedder=embedder, store=store, config=dense_config
            ),
            config=config,
        )
        result = evaluate_retriever(
            retriever,
            queries=dataset.queries,
            qrels=dataset.qrels["train"],
            query_ids=split.validation_query_ids,
            top_k=config.top_k,
        )
    except (
        DataFormatError,
        DatasetValidationError,
        BM25IndexError,
        DenseIndexError,
        HybridRetrievalError,
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
        "retrieval_config": config.model_dump(mode="json"),
        "source_manifests": {
            "bm25": bm25.manifest.model_dump(mode="json"),
            "dense": asdict(snapshot),
        },
        "result": asdict(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
