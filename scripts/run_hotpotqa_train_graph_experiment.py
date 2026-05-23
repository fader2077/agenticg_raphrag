#!/usr/bin/env python
"""Run HotpotQA validation methods against a fixed Neo4j train graph."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.controller import AgenticGraphRAGController
from src.eval.judge_adapter import judge_prediction, load_openai_api_key
from src.eval.run_eval import evaluate_record, evaluate_run_dir, write_method_metrics
from src.generation.answer_generator import AnswerGenerator
from src.indexing.neo4j_hotpotqa_graph import DEFAULT_GRAPH_RUN_ID
from src.io_utils import METHODS, append_jsonl, read_json, read_jsonl, write_json, write_jsonl, write_yaml
from src.indexing.neo4j_hotpotqa_graph import DEFAULT_FULLTEXT_INDEX_NAME
from src.retrieval.neo4j_hotpotqa import Neo4jGraphRetriever, Neo4jHotpotQAStore, Neo4jTextRetriever, Neo4jVectorRetriever


def _run_dir(name: str) -> Path:
    return ROOT / "runs" / name


def _error(qid: str, method: str, stage: str, error_type: str, message: str, recovered: bool, fallback: str) -> dict[str, Any]:
    return {"qid": qid, "method": method, "stage": stage, "error_type": error_type, "error_message": message, "recovered": recovered, "fallback_used": fallback}


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "title": c.get("title"),
            "text": c.get("text"),
            "sentences": c.get("sentences", []),
            "sentence_ids": c.get("sentence_ids", []),
            "score": c.get("score"),
            "rank": c.get("rank"),
            "source": c.get("source"),
        }
        for c in chunks
    ]


def _prediction(method: str, question: dict[str, Any], pred: str, chunks: list[dict[str, Any]], entities: list[Any], paths: list[dict[str, Any]], trace: list[dict[str, Any]], verifier: dict[str, Any], judge: dict[str, Any], cost: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "qid": question.get("qid"),
        "dataset": "hotpotqa",
        "method": method,
        "question": question.get("question"),
        "gold_answer": question.get("answer"),
        "gold_titles": question.get("gold_titles", []),
        "gold_supporting_facts": question.get("supporting_facts", []),
        "pred_answer": pred,
        "retrieved_chunks": _compact_chunks(chunks),
        "retrieved_entities": entities,
        "retrieved_paths": paths,
        "tool_trace": trace,
        "verifier": verifier,
        "judge": judge,
        "metrics": {},
        "cost": cost,
    }
    if extra:
        row.update(extra)
    row["metrics"] = evaluate_record(row)
    return row


def run_methods(
    processed_dir: Path,
    run_name: str,
    graph_run_id: str,
    fulltext_index_name: str,
    methods: list[str],
    eval_size: int | None,
    ablation: str = "full",
) -> dict[str, Any]:
    """Run selected methods against a fixed train graph."""
    questions = read_jsonl(processed_dir / "questions.jsonl")
    if eval_size is not None:
        questions = questions[:eval_size]
    run_root = _run_dir(run_name)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(run_root / "experiment_manifest.json", {}) or {}
    manifest.update(
        {
            "run_name": run_name,
            "graph_run_id": graph_run_id,
            "graph_reuse_policy": "all methods and ablations read this same graph_run_id; no per-ablation graph rebuild",
            "processed_eval_dir": str(processed_dir),
            "eval_split": "validation",
            "eval_question_count": len(questions),
            "methods": methods,
            "qa_model": "deterministic_evidence_sentence_v1",
            "graph_create_model": "deterministic_titlecase_cooccurrence_v2",
            "vectorrag_backend": "neo4j_fulltext_fallback_no_full_train_embeddings",
        }
    )
    write_json(run_root / "experiment_manifest.json", manifest)
    load_openai_api_key()

    for method in methods:
        method_dir = run_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        pred_path = method_dir / "predictions.jsonl"
        err_path = method_dir / "errors.jsonl"
        if pred_path.exists():
            pred_path.unlink()
        if err_path.exists():
            err_path.unlink()
        store = Neo4jHotpotQAStore(graph_run_id=graph_run_id, fulltext_index_name=fulltext_index_name)
        text = Neo4jTextRetriever(store)
        vector = Neo4jVectorRetriever(store)
        graph = Neo4jGraphRetriever(store)
        generator = AnswerGenerator()
        controller = AgenticGraphRAGController(text, vector, graph, {"retrieval": {"top_k_text": 10, "top_k_vector": 10, "top_k_graph_paths": 5, "max_nodes_per_hop": 10}, "agent": {"max_graph_depth": 2, "max_repair_rounds": 1, "verifier_enabled": True}, "generation": {"temperature": 0, "max_context_tokens": 6000}})
        try:
            method_started = time.perf_counter()
            for idx, question in enumerate(questions, start=1):
                qid = str(question.get("qid"))
                started = time.perf_counter()
                chunks: list[dict[str, Any]] = []
                entities: list[Any] = []
                paths: list[dict[str, Any]] = []
                trace: list[dict[str, Any]] = []
                verifier = {"verdict": "not_run", "unsupported_claims": [], "required_missing_evidence": []}
                extra: dict[str, Any] = {"graph_run_id": graph_run_id, "train_graph_used": True}
                try:
                    if method == "LLM-only":
                        gen = generator.generate(str(question.get("question", "")), [], method=method)
                    elif method == "TextRAG":
                        chunks = text.retrieve(str(question.get("question", "")), top_k=10)
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    elif method == "VectorRAG":
                        chunks = vector.retrieve(str(question.get("question", "")), top_k=10)
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    elif method in {"GraphRAG-hop1", "GraphRAG-hop2"}:
                        g = graph.retrieve(str(question.get("question", "")), depth=1 if method == "GraphRAG-hop1" else 2, top_k_paths=5, max_nodes_per_hop=10)
                        chunks = g.get("graph_evidence_chunks", [])
                        entities = g.get("retrieved_entities", [])
                        paths = g.get("retrieved_paths", [])
                        extra["seed_entities"] = g.get("seed_entities", [])
                        extra["retrieved_edges"] = g.get("retrieved_edges", [])
                        if not paths:
                            append_jsonl(err_path, _error(qid, method, "retrieval", "graph_no_path", "No Neo4j train-graph path found.", True, "empty_paths_recorded"))
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    elif method == "AgenticGraphRAG":
                        result = controller.run(qid, str(question.get("question", "")), ablation=ablation)
                        chunks = result.get("retrieved_chunks", [])
                        entities = result.get("retrieved_entities", [])
                        paths = result.get("retrieved_paths", [])
                        trace = result.get("tool_trace", [])
                        verifier = result.get("verifier", verifier)
                        gen = result.get("generation", {"answer": result.get("pred_answer", "insufficient evidence")})
                        extra["retrieved_edges"] = result.get("retrieved_edges", [])
                    else:
                        raise ValueError(f"unknown method: {method}")
                    pred = str(gen.get("answer") or "insufficient evidence")
                    judge = judge_prediction(str(question.get("question", "")), str(question.get("answer", "")), pred, chunks)
                    if judge.get("judge_error"):
                        append_jsonl(err_path, _error(qid, method, "judge", "judge_error", str(judge.get("judge_error")), True, "continue_on_judge_error"))
                    row = _prediction(
                        method,
                        question,
                        pred,
                        chunks,
                        entities,
                        paths,
                        trace,
                        verifier,
                        judge,
                        {"tool_calls": len(trace), "latency_ms": int((time.perf_counter() - started) * 1000), "input_tokens": None, "output_tokens": None},
                        extra,
                    )
                    append_jsonl(pred_path, row)
                except Exception as exc:
                    append_jsonl(err_path, _error(qid, method, "generation", type(exc).__name__, str(exc), True, "insufficient_evidence_prediction"))
                    judge = judge_prediction(str(question.get("question", "")), str(question.get("answer", "")), "insufficient evidence", [])
                    row = _prediction(method, question, "insufficient evidence", [], [], [], trace, {"verdict": "insufficient_evidence", "unsupported_claims": [], "required_missing_evidence": [str(exc)]}, judge, {"tool_calls": len(trace), "latency_ms": int((time.perf_counter() - started) * 1000), "input_tokens": None, "output_tokens": None}, extra)
                    append_jsonl(pred_path, row)
                if idx % 500 == 0 or idx == len(questions):
                    print(f"{method}: {idx}/{len(questions)} predictions elapsed_sec={time.perf_counter() - method_started:.1f}", flush=True)
            if not err_path.exists():
                err_path.touch()
            write_yaml(method_dir / "config_resolved.yaml", manifest)
            write_method_metrics(method_dir)
            print(f"{method}: {len(questions)} predictions")
        finally:
            store.close()
    return evaluate_run_dir(run_root)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed/hotpotqa_validation_full")
    parser.add_argument("--run-name", default="hotpotqa_round2_train_graph_validation")
    parser.add_argument("--graph-run-id", default=DEFAULT_GRAPH_RUN_ID)
    parser.add_argument("--fulltext-index-name", default=DEFAULT_FULLTEXT_INDEX_NAME)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--eval-size", default="full")
    parser.add_argument("--ablation", default="full")
    args = parser.parse_args()
    methods = METHODS if args.all else [args.method]
    if not methods or methods == [None]:
        raise SystemExit("--method or --all required")
    eval_size = None if str(args.eval_size).lower() in {"full", "all", "none"} else int(args.eval_size)
    run_methods(ROOT / args.processed_dir, args.run_name, args.graph_run_id, args.fulltext_index_name, methods, eval_size, args.ablation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
