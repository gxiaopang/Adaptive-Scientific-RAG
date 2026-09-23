"""Validate SciFact and optionally write its deterministic development split manifest."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir or get_settings().scifact_data_dir

    try:
        dataset = load_scifact(data_dir)
        report = validate_scifact(dataset)
        development_split = create_development_split(
            dataset.qrels["train"],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    except (DataFormatError, DatasetValidationError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = {
        "data_dir": str(dataset.data_dir),
        "statistics": asdict(report.statistics),
        "checksums": report.checksums,
        "development_split": asdict(development_split),
    }
    if args.split_output is not None:
        args.split_output.parent.mkdir(parents=True, exist_ok=True)
        args.split_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "statistics": payload["statistics"],
        "checksums": payload["checksums"],
        "development_split": {
            "seed": development_split.seed,
            "validation_fraction": development_split.validation_fraction,
            "calibration_query_count": len(development_split.calibration_query_ids),
            "validation_query_count": len(development_split.validation_query_ids),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
