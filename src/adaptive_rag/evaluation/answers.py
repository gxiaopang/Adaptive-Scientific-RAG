"""Ragas evaluation adapters; optional dependencies are loaded only on demand."""

import math
import os
from collections.abc import Sequence
from importlib import import_module
from threading import Lock
from typing import Any, TypeVar

from pydantic import BaseModel

from adaptive_rag.domain.protocols import EmbeddingProvider
from adaptive_rag.generation.openai_chat import ChatModel

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

Output = TypeVar("Output", bound=BaseModel)
PROMPT_VERSION = "ragas-0.4.3-collections-default"


def make_ragas_embeddings(embedder: EmbeddingProvider) -> Any:
    base = import_module("ragas.embeddings.base").BaseRagasEmbedding

    class Embeddings(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.lock = Lock()

        def embed_text(self, text: str, **kwargs: object) -> list[float]:
            with self.lock:
                return embedder.encode_query(text)

        async def aembed_text(self, text: str, **kwargs: object) -> list[float]:
            return self.embed_text(text)

    return Embeddings()


def finite_score(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        raise ValueError("metric did not return a number")
    return float(value) if math.isfinite(value) else None


class AnswerJudge:
    """Invoke unmodified Ragas metrics and capture structured intermediate results."""

    def __init__(self, model: ChatModel, embeddings: Any) -> None:
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        self.model = model
        self.embeddings = embeddings

    def evaluate(self, question: str, answer: str, context: str) -> dict[str, object]:
        if not question.strip() or not answer.strip() or not context.strip():
            raise ValueError("question, answer and actual context must be nonblank")
        base = import_module("ragas.llms.base").InstructorBaseRagasLLM
        metrics = import_module("ragas.metrics.collections")
        model = self.model
        trace: list[dict[str, object]] = []

        class TracedLLM(base):  # type: ignore[misc, valid-type]
            def generate(self, prompt: str, response_model: type[Output]) -> Output:
                result = model.complete_structured(
                    [{"role": "user", "content": prompt}],
                    response_model,
                )
                trace.append({"schema": response_model.__name__, "output": result.model_dump()})
                return result

            async def agenerate(self, prompt: str, response_model: type[Output]) -> Output:
                return self.generate(prompt, response_model)

        llm = TracedLLM()
        relevance = metrics.AnswerRelevancy(llm=llm, embeddings=self.embeddings, strictness=3)
        faith = metrics.Faithfulness(llm=llm)
        relevance_score = relevance.score(user_input=question, response=answer)
        faith_score = faith.score(
            user_input=question, response=answer, retrieved_contexts=[context]
        )
        return {
            "answer_relevancy": finite_score(relevance_score.value),
            "faithfulness": finite_score(faith_score.value),
            "ragas_trace": trace,
            "undefined_policy": "Ragas NaN is stored as null and excluded with explicit count",
        }


def precision_recall(doc_ids: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    if k < 1 or not relevant or len(set(doc_ids)) != len(doc_ids):
        raise ValueError("invalid retrieval metric inputs")
    hits = len(set(doc_ids[:k]) & relevant)
    return {f"precision@{k}": hits / k, f"recall@{k}": hits / len(relevant)}
