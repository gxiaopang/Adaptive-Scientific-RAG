import json
from pathlib import Path

import pytest

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.retrieval.bm25 import (
    BM25Config,
    BM25IndexError,
    BM25Retriever,
    load_bm25_config,
)

CORPUS_CHECKSUM = "a" * 64


def _corpus() -> dict[str, CorpusDocument]:
    return {
        "d2": CorpusDocument("d2", "Dogs", "Dogs chase cats", {}),
        "d1": CorpusDocument("d1", "Cats", "A cat sits on a mat", {}),
        "d3": CorpusDocument("d3", "Birds", "Birds can fly", {}),
    }


def _config() -> BM25Config:
    return BM25Config(name="test-bm25", k1=0.9, b=0.4, top_k=2)


def test_bm25_retrieves_expected_document() -> None:
    retriever = BM25Retriever.build(
        _corpus(), corpus_checksum=CORPUS_CHECKSUM, config=_config()
    )

    results = retriever.retrieve("mat", top_k=2)

    assert results[0].doc_id == "d1"
    assert [result.rank for result in results] == [1, 2]
    assert all(result.score >= 0.0 for result in results)


def test_bm25_save_load_round_trip(tmp_path: Path) -> None:
    retriever = BM25Retriever.build(
        _corpus(), corpus_checksum=CORPUS_CHECKSUM, config=_config()
    )
    expected = retriever.retrieve("cats", top_k=3)
    index_dir = tmp_path / "index"

    retriever.save(index_dir)
    loaded = BM25Retriever.load(index_dir, mmap=True)

    assert loaded.retrieve("cats", top_k=3) == expected
    assert loaded.manifest.config == _config()
    assert loaded.manifest.corpus_count == 3


def test_bm25_does_not_overwrite_existing_index(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    retriever = BM25Retriever.build(
        _corpus(), corpus_checksum=CORPUS_CHECKSUM, config=_config()
    )

    with pytest.raises(BM25IndexError, match="already exists"):
        retriever.save(index_dir)


def test_bm25_load_rejects_modified_doc_ids(tmp_path: Path) -> None:
    retriever = BM25Retriever.build(
        _corpus(), corpus_checksum=CORPUS_CHECKSUM, config=_config()
    )
    index_dir = tmp_path / "index"
    retriever.save(index_dir)
    (index_dir / "doc_ids.json").write_text(json.dumps(["changed"]), encoding="utf-8")

    with pytest.raises(BM25IndexError, match="corpus count"):
        BM25Retriever.load(index_dir)


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("   ", 1), ("valid", 0)])
def test_bm25_rejects_invalid_retrieval_input(query: str, top_k: int) -> None:
    retriever = BM25Retriever.build(
        _corpus(), corpus_checksum=CORPUS_CHECKSUM, config=_config()
    )

    with pytest.raises(ValueError):
        retriever.retrieve(query, top_k)


def test_load_bm25_config_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bm25.yaml"
    path.write_text(
        "name: invalid\nstrategy: bm25\nk1: 0.9\nb: 0.4\ntop_k: 10\nunknown: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BM25IndexError, match="Unable to load BM25 config"):
        load_bm25_config(path)

