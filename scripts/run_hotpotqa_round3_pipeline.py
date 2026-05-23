#!/usr/bin/env python
"""One-shot Round 3 HotpotQA pipeline with smoke gates, auto main runs, and combined reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.controller import AgenticGraphRAGController
from src.agent.vg_adapter import inspect_vg_environment
from src.data.hotpotqa_loader import load_hotpotqa_rows, sample_rows, write_processed_hotpotqa
from src.eval.judge_adapter import judge_prediction, load_openai_api_key
from src.eval.run_eval import evaluate_record, evaluate_run_dir
from src.generation.answer_generator import AnswerGenerator
from src.indexing.neo4j_hotpotqa_graph import build_hotpotqa_graph_from_chunks
from src.indexing.text_index import TextIndex
from src.indexing.vector_index import VectorIndex
from src.io_utils import METHODS, append_jsonl, load_yaml, read_json, read_jsonl, set_seed, write_json, write_yaml
from src.retrieval.neo4j_hotpotqa import Neo4jGraphRetriever, Neo4jHotpotQAStore
from src.retrieval.text_rag import TextRAGRetriever
from src.retrieval.vector_rag import VectorRAGRetriever


SMOKE_EVAL_MIN = 50
SMOKE_TRAIN_TRANSFER_SIZE = 500


def _hash_payload(payload: Any) -> str:
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _copy_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(cfg))


def _run_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["output"]["run_dir"]


def _processed_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["data"]["processed_base_dir"]


def _text_index_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["indexing"]["text_index"]["dir"]


def _vector_index_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["indexing"]["vector_index"]["dir"]


def _graph_artifact_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["indexing"]["graph_artifact_dir"]


def _stage_cfg(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    out = _copy_cfg(cfg)
    out["runtime_stage"] = stage
    out["output"]["run_dir"] = f"{cfg['output']['run_dir']}_{stage}"
    out["data"]["processed_base_dir"] = f"{cfg['data']['processed_base_dir']}_{stage}"
    out["indexing"]["text_index"]["dir"] = f"{cfg['indexing']['text_index']['dir']}_{stage}"
    out["indexing"]["vector_index"]["dir"] = f"{cfg['indexing']['vector_index']['dir']}_{stage}"
    out["indexing"]["graph_artifact_dir"] = f"{cfg['indexing']['graph_artifact_dir']}_{stage}"
    out["graph"]["graph_run_id"] = f"{cfg['graph']['graph_run_id']}_{stage}"
    out["graph"]["fulltext_index_name"] = f"{cfg['graph']['fulltext_index_name']}_{stage}"
    return out


def _sample_split(
    cfg: dict[str, Any],
    split_role: str,
    split_name: str,
    *,
    fraction: float | None,
    sample_size: int | None,
) -> dict[str, Any]:
    rows, source_meta = load_hotpotqa_rows(cfg["experiment"]["subset"], split_name)
    sample_kwargs: dict[str, Any] = {"seed": int(cfg["experiment"]["random_seed"]), "strategy": str(cfg["data"].get(f"{split_role}_sample_strategy", "random"))}
    if fraction is not None:
        sample_kwargs["sample_fraction"] = fraction
    else:
        sample_kwargs["sample_size"] = sample_size
    sampled, sample_meta = sample_rows(rows, **sample_kwargs)
    output_dir = _processed_root(cfg) / split_role
    manifest = write_processed_hotpotqa(
        output_dir,
        sampled,
        source_meta,
        cfg["experiment"]["subset"],
        sample_meta["resolved_count"],
        int(cfg["experiment"]["random_seed"]),
        split=split_name,
        sample_strategy=sample_meta["sample_strategy"],
        sample_fraction=sample_meta["sample_fraction"],
        sample_ids=sample_meta["sample_ids"],
        max_sentences_per_chunk=int(cfg["indexing"]["chunking"]["max_sentences_per_chunk"]),
        overlap_sentences=int(cfg["indexing"]["chunking"]["overlap_sentences"]),
    )
    manifest["processed_dir"] = str(output_dir)
    manifest["sample_ids_hash"] = _hash_payload(manifest.get("sample_ids", []))
    print(f"[prepare] split_role={split_role} split={split_name} questions={manifest['num_questions']} chunks={manifest['num_chunks']} out={output_dir}", flush=True)
    return manifest


def _prepare_stage_data(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    mode = cfg["experiment"]["benchmark_mode"]
    staged = _stage_cfg(cfg, stage)
    print(f"[prepare] {cfg['experiment']['name']} stage={stage} mode={mode}", flush=True)
    if stage == "smoke":
        eval_size = max(SMOKE_EVAL_MIN, int(round(7405 * 0.01)))
        train_size = SMOKE_TRAIN_TRANSFER_SIZE
    else:
        eval_size = None
        train_size = None
    if mode == "validation_context":
        eval_manifest = _sample_split(
            staged,
            "eval",
            staged["data"]["eval_split"],
            fraction=None if stage == "smoke" else float(staged["data"]["eval_fraction"]),
            sample_size=eval_size if stage == "smoke" else None,
        )
        corpus_manifest = eval_manifest
        manifests = {"corpus": corpus_manifest, "eval": eval_manifest}
    elif mode == "train_graph_transfer":
        train_manifest = _sample_split(
            staged,
            "train",
            staged["data"]["train_split"],
            fraction=None if stage == "smoke" else float(staged["data"]["train_fraction"]),
            sample_size=train_size if stage == "smoke" else None,
        )
        eval_manifest = _sample_split(
            staged,
            "eval",
            staged["data"]["eval_split"],
            fraction=None if stage == "smoke" else float(staged["data"]["eval_fraction"]),
            sample_size=eval_size if stage == "smoke" else None,
        )
        manifests = {"corpus": train_manifest, "train": train_manifest, "eval": eval_manifest}
    else:
        raise ValueError(f"unsupported benchmark_mode: {mode}")
    write_json(_run_root(staged) / "sample_manifest.json", manifests)
    return staged | {"_manifests": manifests}


def _graph_should_rebuild(staged_cfg: dict[str, Any], corpus_manifest: dict[str, Any], graph_manifest_path: Path) -> bool:
    if not staged_cfg["graph"].get("reuse_existing_graph", True):
        return True
    if not graph_manifest_path.exists():
        return True
    existing = read_json(graph_manifest_path, {}) or {}
    current_hash = _hash_payload(
        {
            "graph": staged_cfg["graph"]["extractor"],
            "chunking": staged_cfg["indexing"]["chunking"],
            "corpus_sample_ids_hash": corpus_manifest.get("sample_ids_hash"),
            "benchmark_mode": staged_cfg["experiment"]["benchmark_mode"],
        }
    )
    return staged_cfg["graph"].get("rebuild_if_config_changed", True) and existing.get("graph_config_hash") != current_hash


def _build_indexes_and_graph(staged_cfg: dict[str, Any]) -> dict[str, Any]:
    corpus_manifest = staged_cfg["_manifests"]["corpus"]
    chunks = read_jsonl(Path(corpus_manifest["processed_dir"]) / "chunks.jsonl")
    print(f"[index] {staged_cfg['experiment']['name']} stage={staged_cfg['runtime_stage']} chunks={len(chunks)}", flush=True)
    text_dir = _text_index_dir(staged_cfg)
    vector_dir = _vector_index_dir(staged_cfg)
    graph_dir = _graph_artifact_dir(staged_cfg)
    for path in [text_dir, vector_dir]:
        if path.exists():
            shutil.rmtree(path)
    text_index = TextIndex.build(chunks)
    text_index.save(text_dir)
    print(f"[index] text backend={text_index.backend}", flush=True)
    vector_cfg = staged_cfg["indexing"]["vector_index"]
    vector_index = VectorIndex.build(
        chunks,
        backend=str(vector_cfg["backend"]),
        embed_model=vector_cfg.get("embed_model"),
        ollama_host=staged_cfg["generation"].get("ollama_host"),
        fail_on_error=not bool(vector_cfg.get("fallback_to_fulltext", False)),
    )
    vector_index.save(vector_dir)
    print(f"[index] vector backend={vector_index.backend}", flush=True)
    graph_manifest_path = graph_dir / "run_manifest.json"
    graph_manifest = build_hotpotqa_graph_from_chunks(
        chunks=chunks,
        graph_run_id=staged_cfg["graph"]["graph_run_id"],
        fulltext_index_name=staged_cfg["graph"]["fulltext_index_name"],
        artifact_dir=graph_dir,
        force_rebuild=_graph_should_rebuild(staged_cfg, corpus_manifest, graph_manifest_path),
        batch_size=2000,
        max_edges_per_chunk=2,
        extractor_mode=str(staged_cfg["graph"]["extractor"]["mode"]),
        use_llm_extraction=bool(staged_cfg["graph"]["extractor"].get("use_llm_extraction", False)),
        llm_model=staged_cfg["graph"]["extractor"].get("llm_model"),
        llm_host=staged_cfg["generation"].get("ollama_host"),
        llm_chunk_limit=staged_cfg["graph"]["extractor"].get("max_chunks_for_llm_extraction"),
        extraction_temperature=float(staged_cfg["graph"]["extractor"].get("extraction_temperature", 0.0)),
    )
    graph_manifest["graph_config_hash"] = _hash_payload(
        {
            "graph": staged_cfg["graph"]["extractor"],
            "chunking": staged_cfg["indexing"]["chunking"],
            "corpus_sample_ids_hash": corpus_manifest.get("sample_ids_hash"),
            "benchmark_mode": staged_cfg["experiment"]["benchmark_mode"],
        }
    )
    write_json(graph_manifest_path, graph_manifest)
    print(f"[graph] status={graph_manifest.get('status')} counts={graph_manifest.get('counts')}", flush=True)
    return {
        "text_backend": text_index.backend,
        "vector_backend": vector_index.backend,
        "text_index_dir": str(text_dir),
        "vector_index_dir": str(vector_dir),
        "graph_manifest": graph_manifest,
    }


def _gold_evidence_presence(staged_cfg: dict[str, Any]) -> dict[str, Any]:
    corpus_chunks = read_jsonl(Path(staged_cfg["_manifests"]["corpus"]["processed_dir"]) / "chunks.jsonl")
    eval_questions = read_jsonl(Path(staged_cfg["_manifests"]["eval"]["processed_dir"]) / "questions.jsonl")
    by_title: dict[str, set[int]] = {}
    for chunk in corpus_chunks:
        title = str(chunk.get("title") or "")
        by_title.setdefault(title, set()).update(int(x) for x in chunk.get("sentence_ids", []) if isinstance(x, int) or str(x).isdigit())
    total_titles = 0
    hit_titles = 0
    total_sentences = 0
    hit_sentences = 0
    missing_titles: list[dict[str, Any]] = []
    missing_sentences: list[dict[str, Any]] = []
    for question in eval_questions:
        for title in question.get("gold_titles", []):
            total_titles += 1
            if title in by_title:
                hit_titles += 1
            elif len(missing_titles) < 20:
                missing_titles.append({"qid": question.get("qid"), "title": title})
        for fact in question.get("supporting_facts", []):
            total_sentences += 1
            title = str(fact.get("title") or "")
            sent_id = int(fact.get("sent_id", -1))
            if sent_id in by_title.get(title, set()):
                hit_sentences += 1
            elif len(missing_sentences) < 20:
                missing_sentences.append({"qid": question.get("qid"), "title": title, "sent_id": sent_id, "text": fact.get("text")})
    report = {
        "benchmark_mode": staged_cfg["experiment"]["benchmark_mode"],
        "num_eval_questions": len(eval_questions),
        "total_gold_titles": total_titles,
        "gold_title_presence_rate": hit_titles / total_titles if total_titles else 0.0,
        "total_gold_supporting_sentences": total_sentences,
        "gold_sentence_presence_rate": hit_sentences / total_sentences if total_sentences else 0.0,
        "missing_gold_titles_examples": missing_titles,
        "missing_gold_sentence_examples": missing_sentences,
    }
    write_json(_run_root(staged_cfg) / "gold_evidence_presence_report.json", report)
    return report


def _judge_plan(staged_cfg: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, list[str]]:
    size = int(staged_cfg["judge"].get("openai_judge_sample_size_per_method", 200))
    qids = [str(q.get("qid")) for q in questions]
    plan: dict[str, list[str]] = {}
    for idx, method in enumerate(METHODS):
        import random

        rng = random.Random(int(staged_cfg["experiment"]["random_seed"]) + idx)
        pool = list(qids)
        rng.shuffle(pool)
        plan[method] = pool[: min(size, len(pool))]
    return plan


def _compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in chunks:
        out.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "text": c.get("text"),
                "sentence_ids": c.get("sentence_ids", []),
                "score": c.get("score"),
                "rank": c.get("rank"),
                "source": c.get("source"),
            }
        )
    return out


def _prediction(
    method: str,
    question: dict[str, Any],
    pred_answer: str,
    chunks: list[dict[str, Any]],
    entities: list[Any],
    paths: list[dict[str, Any]],
    tool_trace: list[dict[str, Any]],
    verifier: dict[str, Any],
    judge: dict[str, Any],
    cost: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "qid": question.get("qid"),
        "dataset": "hotpotqa",
        "method": method,
        "question": question.get("question"),
        "gold_answer": question.get("answer"),
        "gold_titles": question.get("gold_titles", []),
        "gold_supporting_facts": question.get("supporting_facts", []),
        "pred_answer": pred_answer,
        "retrieved_chunks": _compact_chunks(chunks),
        "retrieved_entities": entities,
        "retrieved_paths": paths,
        "tool_trace": tool_trace,
        "verifier": verifier,
        "judge": judge,
        "metrics": {},
        "cost": cost,
    }
    row.update(extra)
    row["metrics"] = evaluate_record(row)
    return row


def _run_methods(staged_cfg: dict[str, Any], index_info: dict[str, Any]) -> Path:
    run_root = _run_root(staged_cfg)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[run] {staged_cfg['experiment']['name']} stage={staged_cfg['runtime_stage']} run_root={run_root}", flush=True)
    questions = read_jsonl(Path(staged_cfg["_manifests"]["eval"]["processed_dir"]) / "questions.jsonl")
    judge_plan = _judge_plan(staged_cfg, questions)
    write_json(run_root / "judge_sample_manifest.json", judge_plan)
    manifest = {
        "experiment_name": staged_cfg["experiment"]["name"],
        "benchmark_mode": staged_cfg["experiment"]["benchmark_mode"],
        "dataset_id": staged_cfg["experiment"]["dataset_id"],
        "subset": staged_cfg["experiment"]["subset"],
        "graph_run_id": staged_cfg["graph"]["graph_run_id"],
        "vector_backend": index_info["vector_backend"],
        "text_backend": index_info["text_backend"],
        "graph_manifest": index_info["graph_manifest"],
        "runtime_stage": staged_cfg["runtime_stage"],
        "sample_manifest": staged_cfg["_manifests"],
    }
    write_json(run_root / "experiment_manifest.json", manifest)
    load_openai_api_key(staged_cfg["judge"]["api_key_file"])

    text = TextRAGRetriever(Path(index_info["text_index_dir"]))
    vector = VectorRAGRetriever(Path(index_info["vector_index_dir"]))
    store = Neo4jHotpotQAStore(staged_cfg["graph"]["graph_run_id"], fulltext_index_name=staged_cfg["graph"]["fulltext_index_name"])
    graph = Neo4jGraphRetriever(store)
    generator = AnswerGenerator(
        temperature=float(staged_cfg["generation"]["temperature"]),
        max_context_tokens=int(staged_cfg["generation"]["max_context_tokens"]),
        provider=str(staged_cfg["generation"]["provider"]),
        qa_model=staged_cfg["generation"]["qa_model"],
        ollama_host=staged_cfg["generation"].get("ollama_host"),
        deterministic_fallback_enabled=bool(staged_cfg["generation"]["deterministic_fallback_enabled"]),
        use_deterministic_for_main_eval=bool(staged_cfg["generation"]["use_deterministic_for_main_eval"]),
    )
    controller = AgenticGraphRAGController(text, vector, graph, staged_cfg)
    try:
        for method in METHODS:
            method_dir = run_root / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for name in ["predictions.jsonl", "errors.jsonl", "retrieval_debug.jsonl", "generation_debug.jsonl", "agent_debug.jsonl"]:
                path = method_dir / name
                if path.exists():
                    path.unlink()
            pred_path = method_dir / "predictions.jsonl"
            err_path = method_dir / "errors.jsonl"
            retrieval_debug_path = method_dir / "retrieval_debug.jsonl"
            generation_debug_path = method_dir / "generation_debug.jsonl"
            agent_debug_path = method_dir / "agent_debug.jsonl"
            started_method = time.perf_counter()
            for idx, question in enumerate(questions, start=1):
                qid = str(question.get("qid"))
                started = time.perf_counter()
                chunks: list[dict[str, Any]] = []
                entities: list[Any] = []
                paths: list[dict[str, Any]] = []
                tool_trace: list[dict[str, Any]] = []
                verifier = {"verdict": "not_run", "unsupported_claims": [], "required_missing_evidence": [], "trace": {}}
                extra: dict[str, Any] = {
                    "graph_run_id": staged_cfg["graph"]["graph_run_id"],
                    "generation_provider": staged_cfg["generation"]["provider"],
                    "generation_model": staged_cfg["generation"]["qa_model"],
                    "vector_backend": index_info["vector_backend"],
                }
                try:
                    if method == "LLM-only":
                        gen = generator.generate(str(question.get("question", "")), [], method=method)
                    elif method == "TextRAG":
                        chunks = text.retrieve(str(question.get("question", "")), top_k=int(staged_cfg["retrieval"]["top_k_text"]))
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    elif method == "VectorRAG":
                        chunks = vector.retrieve(str(question.get("question", "")), top_k=int(staged_cfg["retrieval"]["top_k_vector"]))
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    elif method in {"GraphRAG-hop1", "GraphRAG-hop2"}:
                        graph_result = graph.retrieve(
                            str(question.get("question", "")),
                            depth=int(staged_cfg["retrieval"]["graph_hop1_depth"] if method == "GraphRAG-hop1" else staged_cfg["retrieval"]["graph_hop2_depth"]),
                            top_k_paths=int(staged_cfg["retrieval"]["top_k_graph_paths"]),
                            max_nodes_per_hop=int(staged_cfg["retrieval"]["max_nodes_per_hop"]),
                        )
                        chunks = graph_result.get("graph_evidence_chunks", [])
                        entities = graph_result.get("retrieved_entities", [])
                        paths = graph_result.get("retrieved_paths", [])
                        extra["seed_entities"] = graph_result.get("seed_entities", [])
                        extra["retrieved_edges"] = graph_result.get("retrieved_edges", [])
                        extra["path_scores"] = [float(p.get("path_score", 0.0)) for p in paths]
                        gen = generator.generate(str(question.get("question", "")), chunks, method=method)
                    else:
                        result = controller.run(qid, str(question.get("question", "")), ablation="full")
                        chunks = result.get("retrieved_chunks", [])
                        entities = result.get("retrieved_entities", [])
                        paths = result.get("retrieved_paths", [])
                        tool_trace = result.get("tool_trace", [])
                        verifier = result.get("verifier", verifier)
                        gen = result.get("generation", {"answer": result.get("pred_answer", "insufficient evidence")})
                        extra.update(
                            {
                                "seed_entities": result.get("seed_entities", []),
                                "retrieved_edges": result.get("retrieved_edges", []),
                                "pipeline_version": result.get("pipeline_version"),
                                "tools_used": result.get("tools_used", []),
                                "skills_used": result.get("skills_used", []),
                                "repair_trace": result.get("repair_trace", []),
                                "verifier_trace": result.get("verifier_trace", []),
                                "vg_graphrag_integration": result.get("vg_graphrag_integration", {}),
                            }
                        )
                    pred_answer = str(gen.get("answer") or "insufficient evidence")
                    openai_sampled = bool(staged_cfg["judge"].get("openai_judge_enabled", False)) and qid in set(judge_plan[method])
                    judge = judge_prediction(
                        str(question.get("question", "")),
                        str(question.get("answer", "")),
                        pred_answer,
                        chunks,
                        external_binary_enabled=False,
                        external_openai_enabled=openai_sampled,
                    )
                    if judge.get("judge_error"):
                        append_jsonl(err_path, {"qid": qid, "method": method, "stage": "judge", "error_type": "judge_error", "error_message": judge.get("judge_error"), "recovered": True, "fallback_used": "continue_on_judge_error"})
                    generation_trace = {
                        "provider": gen.get("generation_provider"),
                        "model": gen.get("generation_model"),
                        "evidence_count": len(chunks),
                        "raw_generation": gen.get("raw_generation"),
                        "parsed_answer": pred_answer,
                        "fallback_used": bool(gen.get("fallback_used")),
                        "generation_error": gen.get("generation_error"),
                    }
                    extra["generation_trace"] = generation_trace
                    row = _prediction(
                        method,
                        question,
                        pred_answer,
                        chunks,
                        entities,
                        paths,
                        tool_trace,
                        verifier,
                        judge,
                        {
                            "tool_calls": len(tool_trace),
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                            "input_tokens": gen.get("input_tokens"),
                            "output_tokens": gen.get("output_tokens"),
                        },
                        extra,
                    )
                    append_jsonl(pred_path, row)
                    append_jsonl(
                        retrieval_debug_path,
                        {
                            "qid": qid,
                            "method": method,
                            "question": question.get("question"),
                            "gold_answer": question.get("answer"),
                            "gold_titles": question.get("gold_titles", []),
                            "retrieved_titles": [c.get("title") for c in chunks],
                            "gold_title_hit": bool(set(question.get("gold_titles", [])) & {c.get("title") for c in chunks}),
                            "retrieved_chunks_preview": _compact_chunks(chunks[:5]),
                        },
                    )
                    append_jsonl(generation_debug_path, {"qid": qid, "method": method, **generation_trace})
                    if method == "AgenticGraphRAG":
                        append_jsonl(
                            agent_debug_path,
                            {
                                "qid": qid,
                                "pipeline_version": extra.get("pipeline_version"),
                                "tools_used": extra.get("tools_used", []),
                                "skills_used": extra.get("skills_used", []),
                                "vg_graphrag_integration": extra.get("vg_graphrag_integration", {}),
                                "tool_trace": tool_trace,
                                "repair_trace": extra.get("repair_trace", []),
                                "verifier_trace": extra.get("verifier_trace", []),
                            },
                        )
                except Exception as exc:
                    append_jsonl(err_path, {"qid": qid, "method": method, "stage": "generation", "error_type": type(exc).__name__, "error_message": str(exc), "recovered": True, "fallback_used": "insufficient_evidence"})
                    judge = judge_prediction(str(question.get("question", "")), str(question.get("answer", "")), "insufficient evidence", [], external_binary_enabled=False, external_openai_enabled=False)
                    row = _prediction(
                        method,
                        question,
                        "insufficient evidence",
                        [],
                        [],
                        [],
                        tool_trace,
                        {"verdict": "insufficient_evidence", "unsupported_claims": [], "required_missing_evidence": [str(exc)], "trace": {}},
                        judge,
                        {"tool_calls": len(tool_trace), "latency_ms": int((time.perf_counter() - started) * 1000), "input_tokens": None, "output_tokens": None},
                        extra | {
                            "generation_trace": {
                                "provider": staged_cfg["generation"]["provider"],
                                "model": staged_cfg["generation"]["qa_model"],
                                "evidence_count": 0,
                                "raw_generation": None,
                                "parsed_answer": "insufficient evidence",
                                "fallback_used": True,
                                "generation_error": f"{type(exc).__name__}: {exc}",
                            }
                        },
                    )
                    append_jsonl(pred_path, row)
                if idx % 100 == 0 or idx == len(questions):
                    print(f"{staged_cfg['experiment']['name']} {staged_cfg['runtime_stage']} {method}: {idx}/{len(questions)} elapsed_sec={time.perf_counter() - started_method:.1f}", flush=True)
            if not err_path.exists():
                err_path.touch()
            write_yaml(method_dir / "config_resolved.yaml", staged_cfg)
    finally:
        store.close()
    evaluate_run_dir(run_root)
    return run_root


def _smoke_gate(staged_cfg: dict[str, Any], run_root: Path, presence_report: dict[str, Any]) -> dict[str, Any]:
    all_metrics = read_json(run_root / "all_metrics.json", {}) or {}
    reasons: list[str] = []
    warnings: list[str] = []
    eval_count = int(presence_report.get("num_eval_questions", 0))
    if eval_count < SMOKE_EVAL_MIN:
        reasons.append(f"eval sample too small: {eval_count}")
    for method in METHODS:
        method_dir = run_root / method
        if not (method_dir / "predictions.jsonl").exists():
            reasons.append(f"{method} missing predictions.jsonl")
            continue
        records = read_jsonl(method_dir / "predictions.jsonl")
        if not records:
            reasons.append(f"{method} has no predictions")
    agent_dir = run_root / "AgenticGraphRAG"
    agent_rows = read_jsonl(agent_dir / "agent_debug.jsonl")
    if not agent_rows:
        reasons.append("AgenticGraphRAG missing agent_debug.jsonl")
    elif any(not row.get("vg_graphrag_integration") for row in agent_rows[:5]):
        reasons.append("AgenticGraphRAG missing vg_graphrag_integration metadata")
    vector_backend = read_json(run_root / "experiment_manifest.json", {}).get("vector_backend")
    if vector_backend not in {"ollama_embeddings", "sentence_transformers"}:
        reasons.append(f"VectorRAG backend is not dense: {vector_backend}")
    if staged_cfg["experiment"]["benchmark_mode"] == "validation_context":
        if float(presence_report.get("gold_title_presence_rate", 0.0)) < 0.95:
            reasons.append(f"gold_title_presence_rate too low: {presence_report.get('gold_title_presence_rate')}")
        if float(presence_report.get("gold_sentence_presence_rate", 0.0)) < 0.90:
            warnings.append(f"gold_sentence_presence_rate warning: {presence_report.get('gold_sentence_presence_rate')}")
        if max(
            float((all_metrics.get("TextRAG") or {}).get("supporting_fact_recall", 0.0)),
            float((all_metrics.get("VectorRAG") or {}).get("supporting_fact_recall", 0.0)),
        ) <= 0.0:
            reasons.append("TextRAG and VectorRAG supporting_fact_recall are both 0")
    fallback_flags: list[bool] = []
    for method in METHODS:
        for row in read_jsonl(run_root / method / "generation_debug.jsonl"):
            fallback_flags.append(bool(row.get("fallback_used")))
    fallback_rate = (sum(1 for x in fallback_flags if x) / len(fallback_flags)) if fallback_flags else 1.0
    if fallback_rate > 0.20:
        reasons.append(f"generation fallback_used_rate too high: {fallback_rate:.3f}")
    gate = {
        "benchmark_mode": staged_cfg["experiment"]["benchmark_mode"],
        "runtime_stage": staged_cfg["runtime_stage"],
        "passed": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "fallback_used_rate": fallback_rate,
        "vector_backend": vector_backend,
    }
    write_json(run_root / "smoke_gate_result.json", gate)
    return gate


def _write_benchmark_reports(cfg: dict[str, Any], main_run_root: Path, smoke_run_root: Path, title_prefix: str, zh_path: Path, en_path: Path) -> None:
    metrics = read_json(main_run_root / "all_metrics.json", {}) or {}
    smoke_gate = read_json(smoke_run_root / "smoke_gate_result.json", {}) or {}
    presence = read_json(main_run_root / "gold_evidence_presence_report.json", {}) or {}
    manifest = read_json(main_run_root / "experiment_manifest.json", {}) or {}
    integration_sample = {}
    agent_debug = read_jsonl(main_run_root / "AgenticGraphRAG" / "agent_debug.jsonl")
    if agent_debug:
        integration_sample = agent_debug[0].get("vg_graphrag_integration", {})
    lines = [
        "| Method | EM | F1 | SF Precision | SF Recall | Gold Title Recall | Path Found | Path Recall | Avg Tool Calls | Avg Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        m = metrics.get(method, {})
        lines.append(
            f"| {method} | {float(m.get('em', 0.0)):.4f} | {float(m.get('f1', 0.0)):.4f} | {float(m.get('supporting_fact_precision', 0.0)):.4f} | "
            f"{float(m.get('supporting_fact_recall', 0.0)):.4f} | {float(m.get('gold_title_recall', 0.0)):.4f} | "
            f"{float(m.get('path_found_rate', 0.0)):.4f} | {float(m.get('path_recall', 0.0)):.4f} | "
            f"{float(m.get('avg_tool_calls', 0.0)):.2f} | {float(m.get('avg_latency_ms', 0.0)):.1f} |"
        )
    common = [
        "",
        "## Benchmark Mode",
        f"- `{cfg['experiment']['benchmark_mode']}`",
        f"- Smoke gate passed: `{smoke_gate.get('passed')}`",
        f"- Eval sample size: `{manifest.get('sample_manifest', {}).get('eval', {}).get('num_questions')}`",
        f"- Gold title presence rate: `{presence.get('gold_title_presence_rate')}`",
        f"- Gold sentence presence rate: `{presence.get('gold_sentence_presence_rate')}`",
        f"- VectorRAG backend: `{manifest.get('vector_backend')}`",
        "",
        "## Metrics",
        *lines,
        "",
        "## vg_graphragVG Integration",
        f"- integration_status: `{integration_sample.get('integration_status')}`",
        f"- controller_backend: `{integration_sample.get('controller_backend')}`",
        f"- modules_called: `{integration_sample.get('vg_graphrag_modules_called', [])}`",
        f"- adapters_used: `{integration_sample.get('vg_graphrag_adapters_used', [])}`",
        f"- abstractions_embedded: `{integration_sample.get('vg_graphrag_abstractions_embedded', [])}`",
        "",
        "## Debug Artifacts",
        f"- `gold_evidence_presence_report.json`: `{main_run_root / 'gold_evidence_presence_report.json'}`",
        f"- `retrieval_debug.jsonl`: `{main_run_root / 'TextRAG' / 'retrieval_debug.jsonl'}` and per-method equivalents",
        f"- `generation_debug.jsonl`: `{main_run_root / 'VectorRAG' / 'generation_debug.jsonl'}` and per-method equivalents",
        f"- `agent_debug.jsonl`: `{main_run_root / 'AgenticGraphRAG' / 'agent_debug.jsonl'}`",
    ]
    zh_lines = [f"# {title_prefix} 正式報告", ""] + common
    en_lines = [f"# {title_prefix} Report", ""] + common
    zh_path.write_text("\n".join(zh_lines) + "\n", encoding="utf-8")
    en_path.write_text("\n".join(en_lines) + "\n", encoding="utf-8")


def _write_combined_reports(
    val_cfg: dict[str, Any],
    transfer_cfg: dict[str, Any] | None,
    val_main_root: Path,
    transfer_main_root: Path | None,
    zh_path: Path,
    en_path: Path,
) -> None:
    val_metrics = read_json(val_main_root / "all_metrics.json", {}) or {}
    transfer_metrics = read_json(transfer_main_root / "all_metrics.json", {}) if transfer_main_root else {}
    val_presence = read_json(val_main_root / "gold_evidence_presence_report.json", {}) or {}
    transfer_presence = read_json(transfer_main_root / "gold_evidence_presence_report.json", {}) if transfer_main_root else {}
    val_agent = read_jsonl(val_main_root / "AgenticGraphRAG" / "agent_debug.jsonl")
    transfer_agent = read_jsonl(transfer_main_root / "AgenticGraphRAG" / "agent_debug.jsonl") if transfer_main_root else []
    val_int = val_agent[0].get("vg_graphrag_integration", {}) if val_agent else {}
    transfer_int = transfer_agent[0].get("vg_graphrag_integration", {}) if transfer_agent else {}
    lines = [
        "# HotpotQA Round 3 Combined Report",
        "",
        "## Main Benchmark",
        "- `validation_context` is the primary HotpotQA distractor benchmark in this round.",
        f"- Gold title presence rate: `{val_presence.get('gold_title_presence_rate')}`",
        "",
        "## Auxiliary Benchmark",
        "- `train_graph_transfer` is auxiliary cross-split transfer analysis and must not be read as the standard distractor benchmark.",
    ]
    if transfer_main_root:
        lines.append(f"- Gold title presence rate: `{transfer_presence.get('gold_title_presence_rate')}`")
    lines.extend(
        [
            "",
            "## vg_graphragVG Integration Status",
            f"- validation_context: `{val_int.get('integration_status')}` modules={val_int.get('vg_graphrag_modules_called', [])}",
            f"- train_graph_transfer: `{transfer_int.get('integration_status')}` modules={transfer_int.get('vg_graphrag_modules_called', [])}",
            "",
            "## Main Benchmark Metrics",
        ]
    )
    for method in METHODS:
        m = val_metrics.get(method, {})
        lines.append(f"- {method}: EM={float(m.get('em', 0.0)):.4f}, F1={float(m.get('f1', 0.0)):.4f}, SF Recall={float(m.get('supporting_fact_recall', 0.0)):.4f}")
    if transfer_main_root:
        lines.append("")
        lines.append("## Auxiliary Transfer Metrics")
        for method in METHODS:
            m = (transfer_metrics or {}).get(method, {})
            lines.append(f"- {method}: EM={float(m.get('em', 0.0)):.4f}, F1={float(m.get('f1', 0.0)):.4f}, SF Recall={float(m.get('supporting_fact_recall', 0.0)):.4f}")
    zh_path.write_text("\n".join(lines).replace("Combined Report", "綜合報告").replace("Main Benchmark", "主基準").replace("Auxiliary Benchmark", "輔助基準") + "\n", encoding="utf-8")
    en_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_benchmark(cfg: dict[str, Any], *, smoke_then_main: bool, auto_proceed: bool, write_reports: bool) -> tuple[Path, Path]:
    smoke_cfg = _prepare_stage_data(cfg, "smoke")
    index_info = _build_indexes_and_graph(smoke_cfg)
    smoke_run_root = _run_methods(smoke_cfg, index_info)
    presence = _gold_evidence_presence(smoke_cfg)
    gate = _smoke_gate(smoke_cfg, smoke_run_root, presence)
    if not gate["passed"]:
        raise RuntimeError(f"smoke gate failed for {cfg['experiment']['name']}: {gate['reasons']}")
    if not smoke_then_main and not auto_proceed:
        return smoke_run_root, smoke_run_root
    main_cfg = _prepare_stage_data(cfg, "main")
    main_index_info = _build_indexes_and_graph(main_cfg)
    main_run_root = _run_methods(main_cfg, main_index_info)
    _gold_evidence_presence(main_cfg)
    if write_reports:
        _write_benchmark_reports(
            cfg,
            main_run_root,
            smoke_run_root,
            cfg["experiment"]["name"],
            ROOT / cfg["output"]["report_zh"],
            ROOT / cfg["output"]["report_en"],
        )
    return smoke_run_root, main_run_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa_round3_val_context.yaml")
    parser.add_argument("--also-run-transfer", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-then-main", action="store_true")
    parser.add_argument("--auto-proceed", action="store_true")
    parser.add_argument("--run-all-methods", action="store_true")
    parser.add_argument("--integrate-vg-abstractions", action="store_true")
    parser.add_argument("--write-bilingual-reports", action="store_true")
    parser.add_argument("--write-combined-report", action="store_true")
    args = parser.parse_args()

    val_cfg = load_yaml(ROOT / args.config)
    set_seed(int(val_cfg["experiment"]["random_seed"]))
    vg_inspection = inspect_vg_environment(ROOT)
    write_json(ROOT / "reports" / "HOTPOTQA_ROUND3_VG_INSPECTION.json", vg_inspection)
    if args.integrate_vg_abstractions and not vg_inspection.get("integration_possible"):
        raise RuntimeError("vg_graphragVG integration requested but reusable modules are not importable")

    if args.smoke and not args.smoke_then_main:
        smoke_cfg = _prepare_stage_data(val_cfg, "smoke")
        index_info = _build_indexes_and_graph(smoke_cfg)
        smoke_root = _run_methods(smoke_cfg, index_info)
        presence = _gold_evidence_presence(smoke_cfg)
        gate = _smoke_gate(smoke_cfg, smoke_root, presence)
        if args.write_bilingual_reports:
            _write_benchmark_reports(val_cfg, smoke_root, smoke_root, f"{val_cfg['experiment']['name']}_smoke", ROOT / val_cfg["output"]["report_zh"], ROOT / val_cfg["output"]["report_en"])
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0 if gate["passed"] else 1

    smoke_then_main = args.smoke_then_main or args.auto_proceed or args.write_combined_report
    val_smoke_root, val_main_root = _run_benchmark(val_cfg, smoke_then_main=smoke_then_main, auto_proceed=args.auto_proceed, write_reports=args.write_bilingual_reports or True)

    transfer_main_root: Path | None = None
    if args.also_run_transfer:
        transfer_cfg = load_yaml(ROOT / args.also_run_transfer)
        set_seed(int(transfer_cfg["experiment"]["random_seed"]))
        _, transfer_main_root = _run_benchmark(transfer_cfg, smoke_then_main=True, auto_proceed=True, write_reports=args.write_bilingual_reports or True)
    else:
        transfer_cfg = None

    if args.write_combined_report:
        _write_combined_reports(
            val_cfg,
            transfer_cfg,
            val_main_root,
            transfer_main_root,
            ROOT / "reports" / "HOTPOTQA_ROUND3_COMBINED_REPORT_ZH.md",
            ROOT / "reports" / "HOTPOTQA_ROUND3_COMBINED_REPORT_EN.md",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
