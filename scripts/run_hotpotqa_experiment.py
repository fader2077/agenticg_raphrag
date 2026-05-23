#!/usr/bin/env python
"""Run HotpotQA Round 1 methods and write trace-ready predictions."""

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
from src.eval.run_eval import evaluate_record, write_method_metrics
from src.generation.answer_generator import AnswerGenerator
from src.io_utils import (
    METHODS,
    append_jsonl,
    collect_repo_inspection,
    index_dir,
    load_yaml,
    processed_dir,
    read_json,
    read_jsonl,
    run_dir,
    set_seed,
    write_json,
    write_jsonl,
    write_yaml,
)
from src.retrieval.graph_rag import GraphRAGRetriever
from src.retrieval.text_rag import TextRAGRetriever
from src.retrieval.vector_rag import VectorRAGRetriever


def error_row(qid: str, method: str, stage: str, error_type: str, message: str, recovered: bool, fallback: str) -> dict[str, Any]:
    """Create a normalized error row."""
    return {
        "qid": qid,
        "method": method,
        "stage": stage,
        "error_type": error_type,
        "error_message": message,
        "recovered": recovered,
        "fallback_used": fallback,
    }


def compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return retrieved chunks with trace fields retained."""
    out = []
    for chunk in chunks:
        out.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title"),
                "text": chunk.get("text"),
                "sentences": chunk.get("sentences"),
                "sentence_ids": chunk.get("sentence_ids", []),
                "score": chunk.get("score"),
                "rank": chunk.get("rank"),
                "source": chunk.get("source"),
            }
        )
    return out


def build_prediction(
    method: str,
    question: dict[str, Any],
    pred_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    retrieved_entities: list[Any] | None,
    retrieved_paths: list[dict[str, Any]] | None,
    tool_trace: list[dict[str, Any]] | None,
    verifier: dict[str, Any] | None,
    judge: dict[str, Any],
    cost: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the unified prediction schema."""
    record = {
        "qid": question.get("qid"),
        "dataset": "hotpotqa",
        "method": method,
        "question": question.get("question"),
        "gold_answer": question.get("answer"),
        "gold_titles": question.get("gold_titles", []),
        "gold_supporting_facts": question.get("supporting_facts", []),
        "pred_answer": pred_answer,
        "retrieved_chunks": compact_chunks(retrieved_chunks),
        "retrieved_entities": retrieved_entities or [],
        "retrieved_paths": retrieved_paths or [],
        "tool_trace": tool_trace or [],
        "verifier": verifier or {"verdict": "not_run", "unsupported_claims": [], "required_missing_evidence": []},
        "judge": judge,
        "metrics": {},
        "cost": cost,
    }
    if extra:
        record.update(extra)
    record["metrics"] = evaluate_record(record)
    return record


def run_method(method: str, config: dict[str, Any], sample_suffix: str | None, sample_size: int | None, ablation: str) -> dict[str, Any]:
    """Run one method and write all method artifacts."""
    cfg = config
    set_seed(int(cfg.get("experiment", {}).get("random_seed", 42)))
    data_dir = ROOT / processed_dir(cfg, sample_suffix)
    questions = read_jsonl(data_dir / "questions.jsonl")
    if sample_size:
        questions = questions[:sample_size]
    if not questions:
        raise RuntimeError(f"No questions found at {data_dir / 'questions.jsonl'}")

    method_run_dir = ROOT / run_dir(cfg, sample_suffix) / method
    method_run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = method_run_dir / "predictions.jsonl"
    errors_path = method_run_dir / "errors.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()
    if errors_path.exists():
        errors_path.unlink()

    text_retriever = TextRAGRetriever(ROOT / index_dir(cfg, "text_index_dir", sample_suffix)) if method in {"TextRAG", "AgenticGraphRAG"} else None
    vector_retriever = VectorRAGRetriever(ROOT / index_dir(cfg, "vector_index_dir", sample_suffix)) if method in {"VectorRAG", "AgenticGraphRAG"} else None
    graph_retriever = GraphRAGRetriever(ROOT / index_dir(cfg, "graph_index_dir", sample_suffix)) if method in {"GraphRAG-hop1", "GraphRAG-hop2", "AgenticGraphRAG"} else None
    generator = AnswerGenerator(
        temperature=float(cfg.get("generation", {}).get("temperature", 0)),
        max_context_tokens=int(cfg.get("generation", {}).get("max_context_tokens", 6000)),
    )
    controller = (
        AgenticGraphRAGController(text_retriever, vector_retriever, graph_retriever, cfg)
        if method == "AgenticGraphRAG"
        else None
    )
    load_openai_api_key(cfg.get("judge", {}).get("api_key_file"))

    rows: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question.get("qid"))
        start = time.perf_counter()
        try:
            retrieved_chunks: list[dict[str, Any]] = []
            retrieved_entities: list[Any] = []
            retrieved_paths: list[dict[str, Any]] = []
            retrieved_edges: list[dict[str, Any]] = []
            tool_trace: list[dict[str, Any]] = []
            verifier = {"verdict": "not_run", "unsupported_claims": [], "required_missing_evidence": []}
            generation: dict[str, Any]
            extra: dict[str, Any] = {}

            if method == "LLM-only":
                generation = generator.generate(str(question.get("question", "")), [], method=method)
            elif method == "TextRAG":
                retrieved_chunks = text_retriever.retrieve(str(question.get("question", "")), top_k=int(cfg.get("retrieval", {}).get("top_k_text", 10)))
                generation = generator.generate(str(question.get("question", "")), retrieved_chunks, method=method)
            elif method == "VectorRAG":
                retrieved_chunks = vector_retriever.retrieve(str(question.get("question", "")), top_k=int(cfg.get("retrieval", {}).get("top_k_vector", 10)))
                generation = generator.generate(str(question.get("question", "")), retrieved_chunks, method=method)
            elif method in {"GraphRAG-hop1", "GraphRAG-hop2"}:
                depth_key = "graph_depth_hop1" if method == "GraphRAG-hop1" else "graph_depth_hop2"
                graph = graph_retriever.retrieve(
                    str(question.get("question", "")),
                    depth=int(cfg.get("retrieval", {}).get(depth_key, 1 if method == "GraphRAG-hop1" else 2)),
                    top_k_paths=int(cfg.get("retrieval", {}).get("top_k_graph_paths", 5)),
                    max_nodes_per_hop=int(cfg.get("retrieval", {}).get("max_nodes_per_hop", 10)),
                )
                retrieved_chunks = graph.get("graph_evidence_chunks", [])
                retrieved_entities = graph.get("retrieved_entities", [])
                retrieved_paths = graph.get("retrieved_paths", [])
                retrieved_edges = graph.get("retrieved_edges", [])
                extra["seed_entities"] = graph.get("seed_entities", [])
                extra["retrieved_edges"] = retrieved_edges
                if not retrieved_paths:
                    append_jsonl(errors_path, error_row(qid, method, "retrieval", "graph_no_path", "No graph path found.", True, "empty_paths_recorded"))
                generation = generator.generate(str(question.get("question", "")), retrieved_chunks, method=method)
            elif method == "AgenticGraphRAG":
                result = controller.run(qid, str(question.get("question", "")), ablation=ablation)
                retrieved_chunks = result.get("retrieved_chunks", [])
                retrieved_entities = result.get("retrieved_entities", [])
                retrieved_paths = result.get("retrieved_paths", [])
                retrieved_edges = result.get("retrieved_edges", [])
                tool_trace = result.get("tool_trace", [])
                verifier = result.get("verifier", verifier)
                generation = result.get("generation", {"answer": result.get("pred_answer", "insufficient evidence")})
                extra["retrieved_edges"] = retrieved_edges
                extra["repair_used"] = result.get("repair_used", False)
                if not retrieved_paths:
                    append_jsonl(errors_path, error_row(qid, method, "retrieval", "graph_no_path", "No graph path found during agent graph route.", True, "text_vector_fusion_available"))
            else:
                raise ValueError(f"Unknown method: {method}")

            if method != "LLM-only" and not retrieved_chunks:
                append_jsonl(errors_path, error_row(qid, method, "retrieval", "empty_retrieval", "No retrieved chunks.", True, "insufficient_evidence_answer"))
            pred_answer = str(generation.get("answer") or "insufficient evidence")
            judge = judge_prediction(str(question.get("question", "")), str(question.get("answer", "")), pred_answer, retrieved_chunks) if cfg.get("judge", {}).get("enabled", True) else {
                "binary_correct": None,
                "judge_score": None,
                "judge_reason": None,
                "openai_judge_label": None,
                "openai_judge_score": None,
                "openai_judge_reason": None,
                "judge_error": "judge_disabled",
            }
            if judge.get("judge_error"):
                append_jsonl(errors_path, error_row(qid, method, "judge", "judge_error", str(judge.get("judge_error")), True, "continue_on_judge_error"))

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost = {
                "tool_calls": len(tool_trace),
                "latency_ms": latency_ms,
                "input_tokens": generation.get("input_tokens"),
                "output_tokens": generation.get("output_tokens"),
            }
            record = build_prediction(
                method,
                question,
                pred_answer,
                retrieved_chunks,
                retrieved_entities,
                retrieved_paths,
                tool_trace,
                verifier,
                judge,
                cost,
                extra=extra,
            )
            rows.append(record)
            append_jsonl(predictions_path, record)
        except Exception as exc:
            append_jsonl(errors_path, error_row(qid, method, "generation", type(exc).__name__, str(exc), True, "insufficient_evidence_prediction"))
            judge = judge_prediction(str(question.get("question", "")), str(question.get("answer", "")), "insufficient evidence", [])
            record = build_prediction(
                method,
                question,
                "insufficient evidence",
                [],
                [],
                [],
                [],
                {"verdict": "insufficient_evidence", "unsupported_claims": [], "required_missing_evidence": ["exception"]},
                judge,
                {"tool_calls": 0, "latency_ms": int((time.perf_counter() - start) * 1000), "input_tokens": None, "output_tokens": None},
            )
            rows.append(record)
            append_jsonl(predictions_path, record)

    if not errors_path.exists():
        errors_path.touch()
    write_yaml(method_run_dir / "config_resolved.yaml", cfg)
    metrics = write_method_metrics(method_run_dir)
    run_root = ROOT / run_dir(cfg, sample_suffix)
    manifest = read_json(run_root / "experiment_manifest.json", {}) or {}
    manifest.update(
        {
            "experiment": cfg.get("experiment", {}).get("name", "hotpotqa_round1"),
            "sample_suffix": sample_suffix,
            "processed_dir": str(data_dir),
            "methods_requested": cfg.get("experiment", {}).get("methods", METHODS),
            "last_completed_method": method,
            "repo_inspection": manifest.get("repo_inspection") or collect_repo_inspection(),
        }
    )
    write_json(run_root / "experiment_manifest.json", manifest)
    return metrics


def main() -> int:
    """CLI entry point for one or all HotpotQA methods."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", default="configs/hotpotqa_round1.yaml")
    parser.add_argument("--sample-suffix", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--ablation", default="full", choices=["full", "no_verifier", "no_repair", "no_text_route", "no_vector_route", "no_graph_route"])
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    methods = METHODS if args.all else [args.method]
    if not methods or methods == [None]:
        raise SystemExit("--method or --all is required")
    for method in methods:
        metrics = run_method(method, cfg, args.sample_suffix, args.sample_size, args.ablation)
        print(f"{method}: count={metrics.get('count')} em={metrics.get('em')} f1={metrics.get('f1')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
