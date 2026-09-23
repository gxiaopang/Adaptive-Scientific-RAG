import json
from pathlib import Path

import pytest

from adaptive_rag.data.scifact import DataFormatError, load_scifact


def _write_minimal_dataset(root: Path) -> Path:
    data_dir = root / "scifact"
    qrels_dir = data_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (data_dir / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Title", "text": "Body", "metadata": {}})
        + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "Second", "metadata": {}})
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "Claim one", "metadata": {}})
        + "\n"
        + json.dumps({"_id": "q2", "text": "Claim two", "metadata": {}})
        + "\n",
        encoding="utf-8",
    )
    (qrels_dir / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq2\td2\t1\n",
        encoding="utf-8",
    )
    return data_dir


def test_load_scifact_returns_typed_records(tmp_path: Path) -> None:
    dataset = load_scifact(_write_minimal_dataset(tmp_path))

    assert dataset.corpus["d1"].retrieval_text == "Title\n\nBody"
    assert dataset.corpus["d2"].retrieval_text == "Second"
    assert dataset.queries["q1"].text == "Claim one"
    assert dataset.qrels["train"] == {"q1": {"d1": 1}}


def test_loader_reports_duplicate_corpus_id_with_line_number(tmp_path: Path) -> None:
    data_dir = _write_minimal_dataset(tmp_path)
    duplicate = {"_id": "d1", "title": "Again", "text": "Again", "metadata": {}}
    with (data_dir / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")

    with pytest.raises(DataFormatError, match=r"corpus\.jsonl:3: duplicate corpus id 'd1'"):
        load_scifact(data_dir)


def test_loader_rejects_invalid_qrels_header(tmp_path: Path) -> None:
    data_dir = _write_minimal_dataset(tmp_path)
    (data_dir / "qrels" / "train.tsv").write_text(
        "query_id\tcorpus_id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(DataFormatError, match="expected TSV header"):
        load_scifact(data_dir)


def test_loader_rejects_non_positive_relevance(tmp_path: Path) -> None:
    data_dir = _write_minimal_dataset(tmp_path)
    (data_dir / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(DataFormatError, match="score must be positive"):
        load_scifact(data_dir)

