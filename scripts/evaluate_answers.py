"""Evaluate live adaptive RAG on train-derived validation using the .env LLM.

Persists resumable private per-query snapshots plus JSON/Markdown summaries.
No test labels are loaded. Exit 1 means incomplete execution, not a low metric score.
"""

import argparse
import json
import os
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter

from adaptive_rag.core.config import get_settings
from adaptive_rag.data.scifact import load_scifact
from adaptive_rag.data.validation import create_development_split, sha256_file
from adaptive_rag.evaluation.answers import (
    PROMPT_VERSION,
    AnswerJudge,
    make_ragas_embeddings,
    precision_recall,
)
from adaptive_rag.generation.openai_chat import OpenAIChatModel
from adaptive_rag.graph.workflow import AdaptiveRAGWorkflowError
from adaptive_rag.retrieval.dense import load_dense_config
from adaptive_rag.runtime import (
    build_embedding_provider,
    build_postgres_dependencies,
    build_runtime_workflow,
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def error_chain_types(error: BaseException) -> list[str]:
    """Return safe classifications without persisting provider messages or secrets."""

    types = []
    current: BaseException | None = error
    while current is not None and len(types) < 8:
        types.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return types


def main() -> int:
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/evaluations/deepseek_answers_validation")
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke test only; labeled subset")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.limit is not None and args.limit < 1):
        parser.error("workers must be 1–8; limit must be positive")
    settings = get_settings()
    if settings.llm_api_key is None or settings.llm_model is None:
        parser.error("configure ASR_LLM_API_KEY and ASR_LLM_MODEL in .env")
    dataset = load_scifact(settings.scifact_data_dir, splits=("train",))
    split = create_development_split(dataset.qrels["train"])
    ids = list(split.validation_query_ids)
    if args.limit:
        ids = ids[: args.limit]
    source_paths = [
        *sorted(Path("configs").glob("*.yaml")),
        *sorted(Path("src/adaptive_rag").rglob("*.py")),
        Path(__file__),
        settings.scifact_data_dir / "corpus.jsonl",
        settings.scifact_data_dir / "queries.jsonl",
        settings.scifact_data_dir / "qrels/train.tsv",
    ]
    identity = {
        "mode": "live_adaptive_scientific_rag",
        "split": "train-derived-validation",
        "seed": split.seed,
        "validation_fraction": split.validation_fraction,
        "query_ids": ids,
        "full_validation_count": len(split.validation_query_ids),
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "generation_temperature": settings.llm_temperature,
        "generation_max_tokens": settings.llm_max_tokens,
        "thinking_mode": "disabled",
        "judge_temperature": 0.0,
        "judge_max_tokens": 4096,
        "judge_prompt_version": PROMPT_VERSION,
        "answer_relevancy_strictness": 3,
        "judge_framework": "ragas.collections",
        "source_checksums": {str(p): sha256_file(p) for p in source_paths},
        "versions": {name: version(name) for name in ("ragas", "openai", "langgraph", "pydantic")},
        "python": platform.python_version(),
        "machine": platform.node(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o700)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            parser.error("existing evaluation has different inputs; choose a new --output")
    else:
        write_json(manifest_path, identity)
    records = []
    pending = []
    for query_id in ids:
        path = args.output / f"{query_id}.json"
        if path.exists() and (record := json.loads(path.read_text())).get("status") == "succeeded":
            records.append(record)
        else:
            pending.append(query_id)
    resources = None
    postgres = None
    eval_embedder = None
    judge_model = None
    if pending:
        print(f"Loading live retrieval; pending={len(pending)}", flush=True)
        postgres = build_postgres_dependencies(settings)
        resources = build_runtime_workflow(
            settings,
            postgres.sessions,
            training_only=True,
            answer_enable_thinking=False,
        )
        judge_model = OpenAIChatModel(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=0.0,
            max_tokens=4096,
            enable_thinking=False,
        )
        dense_config = load_dense_config(Path("configs/dense.yaml"))
        eval_embedder = build_embedding_provider(settings, dense_config)
        judge = AnswerJudge(judge_model, make_ragas_embeddings(eval_embedder))

        def evaluate_one(query_id: str) -> dict[str, object]:
            path = args.output / f"{query_id}.json"
            record = json.loads(path.read_text()) if path.exists() else {"query_id": query_id}
            started = perf_counter()
            try:
                if "snapshots" not in record:
                    result = resources.workflow.run(dataset.queries[query_id].text)
                    record.update(
                        {
                            "response_mode": result.response_mode,
                            "termination_reason": result.termination_reason,
                            "retrieval_attempts": result.retrieval_attempts,
                            "snapshots": [asdict(s) for s in result.snapshots],
                            "workflow_latency_ms": (perf_counter() - started) * 1000,
                        }
                    )
                    write_json(path, record)
                if not record["snapshots"]:
                    raise ValueError("scientific query routed away from retrieval")
                last = record["snapshots"][-1]
                doc_ids = [d["doc_id"] for d in last["evidence"]]
                relevant = set(dataset.qrels["train"][query_id])
                record["retrieval_metrics"] = {
                    **precision_recall(doc_ids, relevant, 5),
                    **precision_recall(doc_ids, relevant, 10),
                }
                # Judge original request, never the retrieval rewrite.
                record["generation_metrics"] = judge.evaluate(
                    dataset.queries[query_id].text,
                    last["answer"],
                    last["context"],
                )
                record["status"] = "succeeded"
                record.pop("error_type", None)
                record.pop("workflow_error", None)
            except Exception as error:
                record["status"] = "failed"
                record["error_type"] = type(error).__name__
                record["error_chain"] = error_chain_types(error)
                if isinstance(error, AdaptiveRAGWorkflowError):
                    # Workflow messages are controlled, stage-level contract labels.
                    # Provider error text is deliberately not persisted.
                    record["workflow_error"] = str(error)
            record["updated_at"] = datetime.now(UTC).isoformat()
            record["last_execution_ms"] = (perf_counter() - started) * 1000
            write_json(path, record)
            return record

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_one, query_id) for query_id in pending]
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    print(
                        f"{len(records)}/{len(ids)} {record['query_id']} {record['status']}",
                        flush=True,
                    )
        finally:
            if eval_embedder is not None:
                eval_embedder.close()
            if judge_model is not None:
                judge_model.close()
            if resources is not None:
                resources.close()
            if postgres is not None:
                postgres.engine.dispose()
    records.sort(key=lambda r: r["query_id"])
    summary: dict[str, object] = {
        "status": "complete" if all(r["status"] == "succeeded" for r in records) else "incomplete",
        "expected": len(ids),
        "succeeded": sum(r["status"] == "succeeded" for r in records),
        "is_full_validation": len(ids) == len(split.validation_query_ids),
        "failed_query_ids": [r["query_id"] for r in records if r["status"] != "succeeded"],
        "metrics": {},
        "limitations": [
            "Ragas 0.4.3 standard collections metrics; not answer accuracy.",
            "Same configured model generates and judges; no human calibration.",
            "Faithfulness uses only final actual truncated context; no-claim is N/A.",
            "P/R are macro averages against train qrels at final retrieval round.",
            "Relevancy uses 3 reverse questions and the configured online embedding provider.",
            "No quality threshold was prescribed; complete means execution complete.",
        ],
    }
    metrics = {}
    for key in (
        "precision@5",
        "precision@10",
        "recall@5",
        "recall@10",
        "answer_relevancy",
        "faithfulness",
    ):
        group = "retrieval_metrics" if "@" in key else "generation_metrics"
        values = [r[group][key] for r in records if group in r and r[group].get(key) is not None]
        metrics[key] = {
            "mean": mean(values) if values else None,
            "valid_count": len(values),
            "excluded_count": len(ids) - len(values),
        }
    summary["metrics"] = metrics
    write_json(args.output / "summary.json", summary)
    lines = [
        "# DeepSeek 科学 RAG 评估报告",
        "",
        f"执行状态：{summary['status']}",
        f"样本：{summary['succeeded']}/{len(ids)}；完整验证集：{summary['is_full_validation']}",
        f"生成与裁判模型：{settings.llm_model}",
        "",
        "| 指标 | 宏平均 | 有效样本 | 排除样本 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in metrics.items():
        score = "N/A" if item["mean"] is None else f"{item['mean']:.6f}"
        lines.append(f"| {name} | {score} | {item['valid_count']} | {item['excluded_count']} |")
    lines.extend(
        [
            "",
            "Answer Relevancy = Ragas 反向问题与原问题的向量相似度（含非承诺回答判定）；"
            "Faithfulness = 有证据支持的事实主张数 / 全部事实主张数。",
            "Ragas 无定义值为 N/A。分数不是回答准确率。",
            "逐条文件保存每轮实际上下文、答案、Ragas 中间问题、主张与判定理由。",
            "",
            *[f"- {note}" for note in summary["limitations"]],
        ]
    )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
