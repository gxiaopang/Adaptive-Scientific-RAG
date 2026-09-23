"""Deterministic BM25S adapter with versioned, validated index artifacts."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module, metadata
from pathlib import Path
from typing import Literal, Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adaptive_rag.data.scifact import CorpusDocument
from adaptive_rag.domain.models import ScoredDocument


class BM25IndexError(RuntimeError):
    """Raised when a BM25 index cannot be built, loaded, or verified."""


class TokenizerConfig(BaseModel):
    """The intentionally small tokenizer surface supported by the first baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lowercase: bool = True
    stopwords: Literal["none"] = "none"
    stemmer: Literal["none"] = "none"


class BM25Config(BaseModel):
    """Frozen parameters that fully identify a Phase 2 BM25 experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    strategy: Literal["bm25"] = "bm25"
    k1: float = Field(default=0.9, gt=0.0)
    b: float = Field(default=0.4, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1)
    tokenizer: TokenizerConfig = TokenizerConfig()


class BM25IndexManifest(BaseModel):
    """Metadata required to prove that an index matches its corpus and configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    strategy: Literal["bm25"] = "bm25"
    corpus_count: int = Field(gt=0)
    corpus_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    doc_ids_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    bm25s_version: str = Field(min_length=1)
    created_at_utc: datetime
    config: BM25Config


class _Matrix(Protocol):
    def __getitem__(self, key: tuple[int, int]) -> object: ...


class _BM25Model(Protocol):
    def index(
        self,
        corpus: object,
        *,
        show_progress: bool = True,
        leave_progress: bool = False,
    ) -> None: ...

    def retrieve(
        self,
        query_tokens: object,
        *,
        k: int = 10,
        return_as: str = "tuple",
        show_progress: bool = True,
    ) -> tuple[_Matrix, _Matrix]: ...

    def save(self, save_dir: str) -> None: ...


class _BM25Factory(Protocol):
    def __call__(
        self,
        *,
        k1: float,
        b: float,
        method: str,
        backend: str,
    ) -> _BM25Model: ...

    def load(
        self,
        save_dir: str,
        *,
        load_corpus: bool,
        mmap: bool,
    ) -> _BM25Model: ...


class _Tokenizer(Protocol):
    def tokenize(
        self,
        texts: list[str],
        *,
        update_vocab: bool | str = "if_empty",
        show_progress: bool = True,
        return_as: str = "ids",
    ) -> object: ...

    def save_vocab(self, save_dir: str) -> None: ...

    def load_vocab(self, save_dir: str) -> None: ...


class _TokenizerFactory(Protocol):
    def __call__(
        self,
        *,
        lower: bool,
        stopwords: list[str],
        stemmer: None,
    ) -> _Tokenizer: ...


class BM25Retriever:
    """BM25 retriever whose document IDs remain aligned with BEIR qrels."""

    strategy = "bm25"

    def __init__(
        self,
        *,
        model: _BM25Model,
        tokenizer: _Tokenizer,
        doc_ids: Sequence[str],
        manifest: BM25IndexManifest,
    ) -> None:
        if not doc_ids:
            raise ValueError("doc_ids cannot be empty")
        if len(doc_ids) != manifest.corpus_count:
            raise BM25IndexError("Manifest corpus count does not match doc_ids")
        if _doc_ids_checksum(doc_ids) != manifest.doc_ids_checksum:
            raise BM25IndexError("Manifest doc ID checksum does not match doc_ids")
        self._model = model
        self._tokenizer = tokenizer
        self._doc_ids = tuple(doc_ids)
        self.manifest = manifest

    @classmethod
    def build(
        cls,
        corpus: Mapping[str, CorpusDocument],
        *,
        corpus_checksum: str,
        config: BM25Config,
    ) -> "BM25Retriever":
        if not corpus:
            raise ValueError("corpus cannot be empty")
        ordered_documents = sorted(corpus.values(), key=lambda document: document.doc_id)
        doc_ids = tuple(document.doc_id for document in ordered_documents)
        tokenizer = _create_tokenizer(config.tokenizer)
        corpus_tokens = tokenizer.tokenize(
            [document.retrieval_text for document in ordered_documents],
            show_progress=False,
            return_as="tuple",
        )
        model = _bm25_factory()(k1=config.k1, b=config.b, method="lucene", backend="numpy")
        model.index(corpus_tokens, show_progress=False)
        manifest = BM25IndexManifest(
            corpus_count=len(doc_ids),
            corpus_checksum=corpus_checksum,
            doc_ids_checksum=_doc_ids_checksum(doc_ids),
            bm25s_version=metadata.version("bm25s"),
            created_at_utc=datetime.now(UTC),
            config=config,
        )
        return cls(model=model, tokenizer=tokenizer, doc_ids=doc_ids, manifest=manifest)

    @classmethod
    def load(cls, index_dir: Path, *, mmap: bool = False) -> "BM25Retriever":
        try:
            manifest = BM25IndexManifest.model_validate_json(
                (index_dir / "manifest.json").read_text(encoding="utf-8")
            )
            raw_doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise BM25IndexError(
                f"Unable to read BM25 index metadata from {index_dir}: {error}"
            ) from error
        if not isinstance(raw_doc_ids, list) or not all(
            isinstance(doc_id, str) for doc_id in raw_doc_ids
        ):
            raise BM25IndexError("doc_ids.json must contain a list of string IDs")
        doc_ids = cast(list[str], raw_doc_ids)

        try:
            tokenizer = _create_tokenizer(manifest.config.tokenizer)
            tokenizer.load_vocab(str(index_dir))
            model = _bm25_factory().load(str(index_dir), load_corpus=False, mmap=mmap)
        except Exception as error:
            raise BM25IndexError(f"Unable to load BM25S index from {index_dir}: {error}") from error
        return cls(model=model, tokenizer=tokenizer, doc_ids=doc_ids, manifest=manifest)

    def save(self, index_dir: Path) -> None:
        """Atomically save the index without overwriting an existing artifact."""

        if index_dir.exists():
            raise BM25IndexError(f"Index directory already exists: {index_dir}")
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix=f".{index_dir.name}-", dir=index_dir.parent)
        )
        try:
            self._model.save(str(temporary_dir))
            self._tokenizer.save_vocab(str(temporary_dir))
            (temporary_dir / "doc_ids.json").write_text(
                json.dumps(self._doc_ids, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (temporary_dir / "manifest.json").write_text(
                self.manifest.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_dir.rename(index_dir)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    def retrieve(self, query: str, top_k: int) -> list[ScoredDocument]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        result_count = min(top_k, len(self._doc_ids))
        query_tokens = self._tokenizer.tokenize(
            [query],
            update_vocab=False,
            show_progress=False,
            return_as="tuple",
        )
        indices, scores = self._model.retrieve(
            query_tokens,
            k=result_count,
            return_as="tuple",
            show_progress=False,
        )
        candidates = [
            (
                self._doc_ids[cast(int, indices[0, offset])],
                float(cast(float, scores[0, offset])),
            )
            for offset in range(result_count)
        ]
        candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
        return [
            ScoredDocument(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(candidates, start=1)
        ]


def load_bm25_config(path: Path) -> BM25Config:
    try:
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return BM25Config.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise BM25IndexError(f"Unable to load BM25 config from {path}: {error}") from error


def _create_tokenizer(config: TokenizerConfig) -> _Tokenizer:
    return _tokenizer_factory()(lower=config.lowercase, stopwords=[], stemmer=None)


def _doc_ids_checksum(doc_ids: Sequence[str]) -> str:
    serialized_ids = "".join(f"{doc_id}\n" for doc_id in doc_ids)
    return hashlib.sha256(serialized_ids.encode()).hexdigest()


def _bm25_factory() -> _BM25Factory:
    return cast(_BM25Factory, import_module("bm25s").BM25)


def _tokenizer_factory() -> _TokenizerFactory:
    return cast(_TokenizerFactory, import_module("bm25s.tokenization").Tokenizer)
