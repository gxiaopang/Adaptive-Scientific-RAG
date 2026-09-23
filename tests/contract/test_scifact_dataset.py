from pathlib import Path

from adaptive_rag.data.scifact import load_scifact
from adaptive_rag.data.validation import create_development_split, validate_scifact

PROJECT_ROOT = Path(__file__).parents[2]


def test_repository_scifact_dataset_contract() -> None:
    dataset = load_scifact(PROJECT_ROOT / "scifact")
    report = validate_scifact(dataset)
    development_split = create_development_split(dataset.qrels["train"], seed=42)

    assert report.statistics.corpus_count == 5_183
    assert report.statistics.query_count == 1_109
    assert report.statistics.split_query_counts == {"train": 809, "test": 300}
    assert report.statistics.split_qrels_counts == {"train": 919, "test": 339}
    assert len(development_split.calibration_query_ids) == 647
    assert len(development_split.validation_query_ids) == 162
    assert set(development_split.calibration_query_ids).isdisjoint(
        development_split.validation_query_ids
    )
    assert set(development_split.calibration_query_ids) | set(
        development_split.validation_query_ids
    ) == set(dataset.qrels["train"])

