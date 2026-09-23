"""Rank-only reciprocal-rank fusion for the BM25 and dense baselines."""

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from adaptive_rag.domain.models import ScoredDocument
from adaptive_rag.domain.protocols import Retriever


class HybridRetrievalError(RuntimeError):
    """Raised when hybrid configuration or ranked candidates violate the contract."""


class HybridConfig(BaseModel):
    """Frozen parameters that identify the Phase 4 RRF experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    strategy: Literal["hybrid_rrf"] = "hybrid_rrf"
    bm25_candidate_k: int = Field(gt=0)
    dense_candidate_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    top_k: int = Field(gt=0)

    @model_validator(mode="after")
    def candidate_depths_cover_output(self) -> "HybridConfig":
        if self.bm25_candidate_k < self.top_k:
            raise ValueError("bm25_candidate_k must be greater than or equal to top_k")
        if self.dense_candidate_k < self.top_k:
            raise ValueError("dense_candidate_k must be greater than or equal to top_k")
        return self


class HybridRetriever:
    """Retrieve from both independent baselines and fuse their ranks with RRF."""

    strategy = "hybrid_rrf"

    def __init__(
        self,
        *,
        bm25_retriever: Retriever,
        dense_retriever: Retriever,
        config: HybridConfig,
    ) -> None:
        if bm25_retriever.strategy != "bm25":
            raise HybridRetrievalError("bm25_retriever must use the bm25 strategy")
        if dense_retriever.strategy != "dense":
            raise HybridRetrievalError("dense_retriever must use the dense strategy")
        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever
        self.config = config

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if top_k > min(self.config.bm25_candidate_k, self.config.dense_candidate_k):
            raise ValueError("top_k cannot exceed either configured candidate depth")

        bm25_context = copy_context()
        dense_context = copy_context()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as executor:
            bm25_future = executor.submit(
                bm25_context.run,
                self._bm25_retriever.retrieve,
                query,
                self.config.bm25_candidate_k,
            )
            dense_future = executor.submit(
                dense_context.run,
                self._dense_retriever.retrieve,
                query,
                self.config.dense_candidate_k,
            )
            bm25_candidates = bm25_future.result()
            dense_candidates = dense_future.result()
        return reciprocal_rank_fusion(
            (bm25_candidates, dense_candidates),
            rrf_k=self.config.rrf_k,
            top_k=top_k,
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredDocument]],
    *,
    rrf_k: int,
    top_k: int,
) -> list[ScoredDocument]:
    """Fuse ranked lists without comparing their strategy-specific raw scores."""

    if not rankings:
        raise ValueError("rankings cannot be empty")
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    contributions: dict[str, list[float]] = {}
    for ranking_index, ranking in enumerate(rankings):
        if not ranking:
            raise HybridRetrievalError(f"Ranking {ranking_index} cannot be empty")
        seen_doc_ids: set[str] = set()
        for expected_rank, candidate in enumerate(ranking, start=1):
            if not candidate.doc_id:
                raise HybridRetrievalError("Candidate document IDs cannot be empty")
            if candidate.doc_id in seen_doc_ids:
                raise HybridRetrievalError(
                    f"Ranking {ranking_index} contains duplicate document {candidate.doc_id!r}"
                )
            if candidate.rank != expected_rank:
                raise HybridRetrievalError(
                    f"Ranking {ranking_index} must contain consecutive ranks starting at one"
                )
            if not math.isfinite(candidate.score):
                raise HybridRetrievalError(
                    f"Ranking {ranking_index} contains a non-finite raw score"
                )
            contributions.setdefault(candidate.doc_id, []).append(
                1.0 / (rrf_k + candidate.rank)
            )
            seen_doc_ids.add(candidate.doc_id)

    fused = [
        (doc_id, math.fsum(doc_contributions))
        for doc_id, doc_contributions in contributions.items()
    ]
    fused.sort(key=lambda item: (-item[1], item[0]))
    return [
        ScoredDocument(doc_id=doc_id, score=score, rank=rank)
        for rank, (doc_id, score) in enumerate(fused[:top_k], start=1)
    ]


def load_hybrid_config(path: Path) -> HybridConfig:
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return HybridConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise HybridRetrievalError(f"Unable to load hybrid config from {path}: {error}") from error
