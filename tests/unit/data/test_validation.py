from pathlib import Path

import pytest

from adaptive_rag.data.scifact import CorpusDocument, Query, SciFactDataset
from adaptive_rag.data.validation import (
    DatasetValidationError,
    create_development_split,
    validate_scifact,
)


def _dataset(data_dir: Path) -> SciFactDataset:
    return SciFactDataset(
        data_dir=data_dir,
        corpus={
            "d1": CorpusDocument("d1", "One", "Text", {}),
            "d2": CorpusDocument("d2", "Two", "Text", {}),
        },
        queries={
            "q1": Query("q1", "First", {}),
            "q2": Query("q2", "Second", {}),
        },
        qrels={"train": {"q1": {"d1": 1}}, "test": {"q2": {"d2": 1}}},
    )


def _write_source_files(data_dir: Path) -> None:
    (data_dir / "qrels").mkdir(parents=True)
    (data_dir / "corpus.jsonl").write_text("corpus\n", encoding="utf-8")
    (data_dir / "queries.jsonl").write_text("queries\n", encoding="utf-8")
    (data_dir / "qrels" / "train.tsv").write_text("train\n", encoding="utf-8")
    (data_dir / "qrels" / "test.tsv").write_text("test\n", encoding="utf-8")


def test_validate_scifact_returns_statistics_and_checksums(tmp_path: Path) -> None:
    data_dir = tmp_path / "scifact"
    _write_source_files(data_dir)

    report = validate_scifact(_dataset(data_dir))

    assert report.statistics.corpus_count == 2
    assert report.statistics.query_count == 2
    assert report.statistics.split_query_counts == {"train": 1, "test": 1}
    assert report.statistics.split_qrels_counts == {"train": 1, "test": 1}
    assert set(report.checksums) == {
        "corpus.jsonl",
        "queries.jsonl",
        "qrels/train.tsv",
        "qrels/test.tsv",
    }
    assert all(len(checksum) == 64 for checksum in report.checksums.values())


def test_validate_scifact_collects_cross_file_issues(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    dataset.qrels["test"] = {"q1": {"missing-doc": 1}, "missing-query": {"d2": 1}}

    with pytest.raises(DatasetValidationError) as raised:
        validate_scifact(dataset)

    message = str(raised.value)
    assert "overlaps earlier splits" in message
    assert "unknown queries" in message
    assert "unknown documents" in message
    assert "queries have no qrels" in message


def test_development_split_is_deterministic_disjoint_and_complete() -> None:
    train_qrels = {
        f"q{query_index}": {
            f"d{query_index}-{document_index}": 1
            for document_index in range(1 + query_index % 3)
        }
        for query_index in range(100)
    }

    first = create_development_split(train_qrels, validation_fraction=0.2, seed=42)
    repeated = create_development_split(train_qrels, validation_fraction=0.2, seed=42)
    another_seed = create_development_split(train_qrels, validation_fraction=0.2, seed=43)

    calibration_ids = set(first.calibration_query_ids)
    validation_ids = set(first.validation_query_ids)
    assert first == repeated
    assert first.validation_query_ids != another_seed.validation_query_ids
    assert len(calibration_ids) == 80
    assert len(validation_ids) == 20
    assert calibration_ids.isdisjoint(validation_ids)
    assert calibration_ids | validation_ids == set(train_qrels)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_development_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        create_development_split({"q1": {"d1": 1}, "q2": {"d2": 1}}, validation_fraction=fraction)

