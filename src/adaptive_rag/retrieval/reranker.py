"""Online reranking over a bounded Hybrid RRF candidate set."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.domain.protocols import PairwiseReranker, Retriever


class RerankerError(RuntimeError):
    """Raised when reranker configuration, candidates, or output is invalid."""


class RerankerConfig(BaseModel):
    """Frozen online reranker experiment parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    strategy: Literal["hybrid_rrf_reranker"] = "hybrid_rrf_reranker"
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    candidate_k: int = Field(gt=0)
    top_k: int = Field(gt=0)

    @model_validator(mode="after")
    def candidate_depth_covers_output(self) -> "RerankerConfig":
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class RerankingRetriever:
    """Resolve Hybrid candidates to corpus text and rerank them online."""

    strategy = "hybrid_rrf_reranker"

    def __init__(
        self,
        *,
        candidate_retriever: Retriever,
        reranker: PairwiseReranker,
        corpus: Mapping[str, CorpusDocument],
        config: RerankerConfig,
    ) -> None:
        if candidate_retriever.strategy != "hybrid_rrf":
            raise RerankerError("candidate_retriever must use the hybrid_rrf strategy")
        if reranker.model_name != config.model_name or reranker.provider != config.provider:
            raise RerankerError("Loaded reranker provider does not match configuration")
        if not corpus:
            raise ValueError("corpus cannot be empty")
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._corpus = corpus
        self.config = config

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if top_k > self.config.candidate_k:
            raise ValueError("top_k cannot exceed candidate_k")

        candidates = self._candidate_retriever.retrieve(query, self.config.candidate_k)
        self._validate_candidates(candidates)
        documents: list[str] = []
        for candidate in candidates:
            try:
                documents.append(self._corpus[candidate.doc_id].retrieval_text)
            except KeyError as error:
                raise RerankerError(
                    f"Candidate document {candidate.doc_id!r} is missing from the corpus"
                ) from error
        scores = self._reranker.score(query, documents)
        if len(scores) != len(candidates):
            raise RerankerError(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        if not all(math.isfinite(score) for score in scores):
            raise RerankerError("Reranker returned a non-finite score")

        rescored = list(zip(candidates, scores, strict=True))
        rescored.sort(key=lambda item: (-item[1], item[0].rank, item[0].doc_id))
        return [
            ScoredDocument(doc_id=candidate.doc_id, score=score, rank=rank)
            for rank, (candidate, score) in enumerate(rescored[:top_k], start=1)
        ]

    @staticmethod
    def _validate_candidates(candidates: Sequence[ScoredDocument]) -> None:
        if not candidates:
            raise RerankerError("Candidate retriever returned no documents")
        seen_doc_ids: set[str] = set()
        for expected_rank, candidate in enumerate(candidates, start=1):
            if candidate.rank != expected_rank:
                raise RerankerError("Candidates must contain consecutive ranks starting at one")
            if not candidate.doc_id:
                raise RerankerError("Candidate document IDs cannot be empty")
            if candidate.doc_id in seen_doc_ids:
                raise RerankerError(f"Duplicate candidate document {candidate.doc_id!r}")
            if not math.isfinite(candidate.score):
                raise RerankerError("Candidate scores must be finite")
            seen_doc_ids.add(candidate.doc_id)


def load_reranker_config(path: Path) -> RerankerConfig:
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return RerankerConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise RerankerError(f"Unable to load reranker config from {path}: {error}") from error
