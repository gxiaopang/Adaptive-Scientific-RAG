"""Build or reuse the SciFact online-embedding pgvector index."""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from adaptive_rag.core.config import get_settings
from adaptive_rag.data.scifact import DataFormatError, load_scifact
from adaptive_rag.data.validation import DatasetValidationError, validate_scifact
from adaptive_rag.domain.protocols import ProviderInvocationError
from adaptive_rag.retrieval.dense import DenseIndexError, build_dense_index, load_dense_config
from adaptive_rag.runtime import build_embedding_provider, build_postgres_dependencies
from adaptive_rag.storage.postgres.vector_store import PgVectorDenseStore, PgVectorStoreError

DEFAULT_CONFIG = Path("configs/dense.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    data_dir = args.data_dir or settings.scifact_data_dir
    postgres = None
    embedder = None
    try:
        dataset = load_scifact(data_dir)
        report = validate_scifact(dataset)
        config = load_dense_config(args.config)
        postgres = build_postgres_dependencies(settings)
        embedder = build_embedding_provider(settings, config)
        store = PgVectorDenseStore(
            postgres.sessions,
            index_name=config.index_name,
            dimension=config.dimension,
        )
        before = None
        try:
            before = store.snapshot()
        except PgVectorStoreError:
            pass
        snapshot = build_dense_index(
            dataset.corpus,
            corpus_checksum=report.checksums["corpus.jsonl"],
            config=config,
            embedder=embedder,
            store=store,
        )
        reused = before == snapshot
    except (
        DataFormatError,
        DatasetValidationError,
        DenseIndexError,
        PgVectorStoreError,
        ProviderInvocationError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if embedder is not None:
            embedder.close()
        if postgres is not None:
            postgres.engine.dispose()

    print(
        json.dumps(
            {"index": asdict(snapshot), "reused": reused, "config": config.model_dump(mode="json")},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
