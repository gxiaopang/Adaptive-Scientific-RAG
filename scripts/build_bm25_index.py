"""Build and persist the Phase 2 SciFact BM25 index."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from adaptive_rag.core.config import get_settings
from adaptive_rag.data.scifact import DataFormatError, load_scifact
from adaptive_rag.data.validation import DatasetValidationError, validate_scifact
from adaptive_rag.retrieval.bm25 import BM25IndexError, BM25Retriever, load_bm25_config

DEFAULT_CONFIG = Path("configs/bm25.yaml")
DEFAULT_INDEX_DIR = Path(".artifacts/indexes/bm25/scifact-v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or get_settings().scifact_data_dir
    try:
        dataset = load_scifact(data_dir)
        validation_report = validate_scifact(dataset)
        config = load_bm25_config(args.config)
        retriever = BM25Retriever.build(
            dataset.corpus,
            corpus_checksum=validation_report.checksums["corpus.jsonl"],
            config=config,
        )
        retriever.save(args.index_dir)
    except (DataFormatError, DatasetValidationError, BM25IndexError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "index_dir": str(args.index_dir.resolve()),
                "strategy": retriever.strategy,
                "corpus_count": retriever.manifest.corpus_count,
                "corpus_checksum": retriever.manifest.corpus_checksum,
                "config": retriever.manifest.config.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

