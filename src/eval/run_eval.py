"""Prediction evaluation, aggregation, and report helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.eval.agent_metrics import agent_case_metrics
from src.eval.answer_metrics import exact_match, normalized_accuracy, token_f1
from src.eval.graph_metrics import graph_case_metrics
from src.eval.retrieval_metrics import chunk_recall_at_k, mrr, ndcg
from src.eval.supporting_fact_metrics import gold_title_metrics, supporting_fact_metrics
from src.io_utils import METHODS, read_json, read_jsonl, write_json, write_jsonl


def evaluate_record(record: dict[str, Any]) -> dict[str, float]:
    """Compute deterministic per-record answer, retrieval, graph, and agent metrics."""
    pred = record.get("pred_answer", "")
    gold = record.get("gold_answer", "")
    chunks = record.get("retrieved_chunks", [])
    gold_facts = record.get("gold_supporting_facts", [])
    gold_titles = record.get("gold_titles") or sorted({f.get("title") for f in gold_facts if f.get("title")})
    metrics: dict[str, float] = {
        "em": float(exact_match(pred, gold)),
        "f1": float(token_f1(pred, gold)),
        "normalized_answer_accuracy": float(normalized_accuracy(pred, gold)),
        "chunk_recall@5": chunk_recall_at_k(gold_titles, chunks, 5),
        "chunk_recall@10": chunk_recall_at_k(gold_titles, chunks, 10),
        "chunk_recall@20": chunk_recall_at_k(gold_titles, chunks, 20),
        "sentence_recall@5": supporting_fact_metrics(gold_facts, chunks[:5]).get("supporting_fact_recall", 0.0),
        "sentence_recall@10": supporting_fact_metrics(gold_facts, chunks[:10]).get("supporting_fact_recall", 0.0),
        "sentence_recall@20": supporting_fact_metrics(gold_facts, chunks[:20]).get("supporting_fact_recall", 0.0),
        "mrr": mrr(gold_titles, chunks),
        "ndcg": ndcg(gold_titles, chunks, 10),
    }
    metrics.update(supporting_fact_metrics(gold_facts, chunks))
    metrics.update(gold_title_metrics(gold_titles, chunks))
    metrics.update(graph_case_metrics({**record, "gold_titles": gold_titles}))
    metrics.update(agent_case_metrics(record))
    return metrics


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all formal metrics over prediction records."""
    if not records:
        return {"count": 0}
    per_case = [evaluate_record(record) for record in records]
    keys = sorted({key for item in per_case for key in item})
    metrics: dict[str, Any] = {"count": len(records)}
    for key in keys:
        vals = [float(item[key]) for item in per_case if item.get(key) is not None]
        metrics[key] = sum(vals) / len(vals) if vals else 0.0

    judge_rows = [r.get("judge", {}) for r in records]
    binary_vals = [j.get("judge_score") for j in judge_rows if j.get("judge_score") is not None]
    openai_vals = [j.get("openai_judge_score") for j in judge_rows if j.get("openai_judge_score") is not None]
    labels = [j.get("openai_judge_label") for j in judge_rows if j.get("openai_judge_label")]
    metrics["judge_binary_accuracy"] = sum(float(x) for x in binary_vals) / len(binary_vals) if binary_vals else None
    metrics["openai_judge_score_mean"] = sum(float(x) for x in openai_vals) / len(openai_vals) if openai_vals else None
    metrics["openai_judge_accuracy"] = labels.count("correct") / len(labels) if labels else None
    metrics["judge_error_rate"] = sum(1 for j in judge_rows if j.get("judge_error")) / len(judge_rows) if judge_rows else 0.0

    costs = [r.get("cost", {}) for r in records]
    metrics["avg_latency_ms"] = sum(float(c.get("latency_ms") or 0) for c in costs) / len(costs)
    metrics["total_latency_ms"] = sum(float(c.get("latency_ms") or 0) for c in costs)
    metrics["total_tool_calls"] = sum(float(c.get("tool_calls") or 0) for c in costs)
    metrics["avg_tool_calls"] = metrics["total_tool_calls"] / len(records)
    correct = max(1.0, sum(float(m.get("em", 0.0)) for m in per_case))
    metrics["cost_per_correct_answer"] = metrics["total_tool_calls"] / correct
    return metrics


def write_method_metrics(method_dir: Path) -> dict[str, Any]:
    """Evaluate one method directory and write method-level metric files."""
    records = read_jsonl(method_dir / "predictions.jsonl")
    for record in records:
        record.setdefault("metrics", {}).update(evaluate_record(record))
    metrics = aggregate(records)
    retrieval_metrics = {
        k: v
        for k, v in metrics.items()
        if "recall" in k or k in {"mrr", "ndcg", "gold_title_precision", "gold_title_f1", "gold_title_recall"}
    }
    judge_metrics = {k: v for k, v in metrics.items() if "judge" in k}
    write_json(method_dir / "metrics.json", metrics)
    write_json(method_dir / "retrieval_metrics.json", retrieval_metrics)
    write_json(method_dir / "judge_metrics.json", judge_metrics)
    write_jsonl(method_dir / "predictions.jsonl", records)
    return metrics


def evaluate_run_dir(run_dir: Path) -> dict[str, Any]:
    """Evaluate every available method under a run directory."""
    all_metrics: dict[str, Any] = {}
    for method in METHODS:
        method_dir = run_dir / method
        if (method_dir / "predictions.jsonl").exists():
            all_metrics[method] = write_method_metrics(method_dir)
    write_json(run_dir / "all_metrics.json", all_metrics)
    write_summary_csv(run_dir, all_metrics)
    write_error_analysis(run_dir)
    write_summary_md(run_dir, all_metrics)
    manifest = read_json(run_dir / "experiment_manifest.json", {}) or {}
    manifest["evaluated_methods"] = list(all_metrics)
    write_json(run_dir / "experiment_manifest.json", manifest)
    return all_metrics


def write_summary_csv(run_dir: Path, all_metrics: dict[str, Any]) -> None:
    """Write a compact method-comparison CSV."""
    keys = [
        "method",
        "count",
        "em",
        "f1",
        "normalized_answer_accuracy",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
        "supporting_fact_recall@5",
        "supporting_fact_recall@10",
        "supporting_fact_recall@20",
        "gold_title_precision",
        "gold_title_recall",
        "gold_title_f1",
        "chunk_recall@5",
        "chunk_recall@10",
        "chunk_recall@20",
        "mrr",
        "ndcg",
        "path_found",
        "path_found_rate",
        "path_recall",
        "avg_path_score",
        "entity_link_success_rate",
        "avg_tool_calls",
        "repair_used",
        "repair_success",
        "verifier_unsupported",
        "final_unsupported",
        "judge_binary_accuracy",
        "openai_judge_score_mean",
        "openai_judge_accuracy",
        "judge_error_rate",
        "avg_latency_ms",
        "total_latency_ms",
        "total_tool_calls",
        "cost_per_correct_answer",
    ]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, method_metrics in all_metrics.items():
            row = {"method": method}
            row.update({key: method_metrics.get(key) for key in keys if key != "method"})
            writer.writerow(row)


def write_error_analysis(run_dir: Path) -> None:
    """Merge per-method errors into error_analysis.csv."""
    rows = []
    for method in METHODS:
        for item in read_jsonl(run_dir / method / "errors.jsonl"):
            rows.append(item)
    keys = ["qid", "method", "stage", "error_type", "error_message", "recovered", "fallback_used"]
    with (run_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_summary_md(run_dir: Path, all_metrics: dict[str, Any]) -> None:
    """Write a Markdown summary table."""
    lines = [
        "# HotpotQA Round 1 Summary",
        "",
        "| Method | Count | EM | F1 | SF F1 | Gold Title Recall | Path Found | Judge Error Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in all_metrics.items():
        lines.append(
            f"| {method} | {m.get('count', 0)} | {m.get('em', 0):.4f} | {m.get('f1', 0):.4f} | "
            f"{m.get('supporting_fact_f1', 0):.4f} | {m.get('gold_title_recall', 0):.4f} | "
            f"{m.get('path_found', 0):.4f} | {m.get('judge_error_rate', 0):.4f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(run_dir: Path, report_path: Path, all_metrics: dict[str, Any], smoke_passed: bool | None = None) -> None:
    """Write the full benchmark report requested by the pipeline."""
    manifest = read_json(run_dir / "experiment_manifest.json", {}) or {}
    lines = [
        "# HotpotQA Round 1 Report",
        "",
        "## 1. Experiment Setting",
        f"- Run directory: `{run_dir}`",
        "- Dataset: HotpotQA distractor validation/dev when available.",
        "- Methods: LLM-only, TextRAG, VectorRAG, GraphRAG-hop1, GraphRAG-hop2, AgenticGraphRAG.",
        "- Shared answer style: evidence-only concise answer; LLM-only uses a question-only prompt.",
        "- Temperature: 0.",
        "- Max context budget: 6000 tokens.",
        f"- Smoke test passed: {smoke_passed if smoke_passed is not None else manifest.get('smoke_passed', 'not recorded')}.",
        f"- Full run completed: {manifest.get('full_run_completed', report_path.name.upper().find('FULL') >= 0)}.",
        "",
        "## 2. Dataset And Sample",
        f"- Sample suffix: {manifest.get('sample_suffix')}.",
        f"- Processed data directory: `{manifest.get('processed_dir', 'not recorded')}`.",
        "- The loader normalizes HotpotQA rows into questions, documents, sentence-aware chunks, supporting facts, and gold titles.",
        "",
        "## 3. Method Definitions",
        "- LLM-only: answers from the question-only prompt with no retrieval context.",
        "- TextRAG: BM25 or TF-IDF text search over HotpotQA chunks, then shared evidence-only generation.",
        "- VectorRAG: dense/vector-style search with sentence-transformer support and a local TF-IDF vector fallback.",
        "- GraphRAG-hop1: entity linking, one-hop graph expansion, provenance-linked chunks, shared generation.",
        "- GraphRAG-hop2: entity linking, two-hop path enumeration/ranking, path provenance, shared generation.",
        "- AgenticGraphRAG: bounded deterministic controller with text, vector, graph, fusion, verification, and one repair round.",
        "",
        "## 4. Index Construction",
        "- Text index: rank_bm25 when installed, otherwise sklearn TF-IDF.",
        "- Vector index: sentence-transformers plus FAISS when explicitly enabled; default local char n-gram vector fallback.",
        "- Graph index: rule-based entity/relation extraction with mandatory edge provenance, persisted through NetworkX/JSON.",
        "",
        "## 5. GraphRAG-hop1 Versus VectorRAG",
        "GraphRAG-hop1 explicitly links question mentions to graph nodes, expands relation neighborhoods, and only admits chunks attached to provenance-carrying graph edges. VectorRAG ranks chunks directly by vector similarity without entity linking or graph traversal.",
        "",
        "## 6. GraphRAG-hop2 Path Retrieval",
        "GraphRAG-hop2 enumerates one- and two-edge paths from linked seed nodes, ranks paths by question/path token overlap, and uses edge provenance to collect evidence chunks. The prediction trace includes `retrieved_paths` even when no path is found.",
        "",
        "## 7. AgenticGraphRAG Tool Flow",
        "`analyze_question -> text_search -> vector_search -> graph_expand -> evidence_fuse -> answer_generate -> evidence_verify -> optional repair_retrieve -> answer_generate -> evidence_verify`",
        "",
        "## 8. Metrics Table",
        "| Method | EM | F1 | SF Precision | SF Recall | SF F1 | Gold Title Recall | Path Found | Path Recall | Tool Calls | Repair | Judge Binary | OpenAI Judge | Judge Error | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, m in all_metrics.items():
        lines.append(
            f"| {method} | {m.get('em', 0):.4f} | {m.get('f1', 0):.4f} | "
            f"{m.get('supporting_fact_precision', 0):.4f} | {m.get('supporting_fact_recall', 0):.4f} | "
            f"{m.get('supporting_fact_f1', 0):.4f} | {m.get('gold_title_recall', 0):.4f} | "
            f"{m.get('path_found', 0):.4f} | {m.get('path_recall', 0):.4f} | "
            f"{m.get('avg_tool_calls', 0):.2f} | {m.get('repair_used', 0):.4f} | "
            f"{_fmt_nullable(m.get('judge_binary_accuracy'))} | {_fmt_nullable(m.get('openai_judge_score_mean'))} | "
            f"{m.get('judge_error_rate', 0):.4f} | {m.get('avg_latency_ms', 0):.1f} |"
        )

    lines.extend(
        [
            "",
            "## 9. judge_binary_correctness.py Results",
            "Integrated through `src/eval/judge_adapter.py`. The adapter imports the repository judge when external calls are enabled and otherwise records a deterministic local fallback result without exposing the API key.",
            "",
            "## 10. judge_openai.py Results",
            "Integrated through `src/eval/judge_adapter.py`. OpenAI judge failures are converted into `judge_error` fields and do not stop EM/F1/retrieval evaluation.",
            "",
            "## 11. Error Analysis",
            "Structured errors are written to `error_analysis.csv`. Categories include retrieval_miss, partial_evidence, wrong_entity_link, graph_no_path, noisy_graph, unsupported_answer, judge_error, generation_error, and over_repair.",
            "",
            "## 12. Failure Cases",
        ]
    )
    failures = collect_failures(run_dir, limit=10)
    if failures:
        for idx, case in enumerate(failures, start=1):
            m = case.get("metrics") or evaluate_record(case)
            lines.append(
                f"{idx}. `{case.get('method')}` `{case.get('qid')}` EM={m.get('em', 0):.0f} "
                f"F1={m.get('f1', 0):.3f}; pred=`{case.get('pred_answer')}`; gold=`{case.get('gold_answer')}`"
            )
    else:
        lines.append("- No failed cases found in available predictions.")

    lines.extend(["", "## 13. AgenticGraphRAG Versus GraphRAG-hop2 Cases"])
    comparisons = compare_agent_vs_graph(run_dir, limit=10)
    if comparisons:
        for idx, row in enumerate(comparisons, start=1):
            direction = "improved" if row["delta"] > 0 else "regressed" if row["delta"] < 0 else "tied"
            lines.append(
                f"{idx}. `{row['qid']}` {direction}: GraphRAG-hop2 F1={row['graph_f1']:.3f}, "
                f"AgenticGraphRAG F1={row['agent_f1']:.3f}, delta={row['delta']:.3f}"
            )
    else:
        lines.append("- No overlapping cases were available for comparison.")

    lines.extend(
        [
            "",
            "## 14. Current Limitations",
            "- Default answer generation is deterministic and evidence-only for reproducibility; it is a runnable fallback, not a high-quality answer model.",
            "- Default external OpenAI judge calls are controlled by `HOTPOTQA_RUN_OPENAI_JUDGE_CALLS=1` to avoid accidental high-volume judge traffic.",
            "- Rule-based graph extraction is provenance-safe but relation semantics are shallow.",
            "- Neo4j is optional; the completed run uses the local NetworkX/JSON graph store when Neo4j is unavailable.",
            "",
            "## 15. Next Steps",
            "- Enable batched external judge calls for a final paper-grade scoring run.",
            "- Replace rule-based relation extraction with a HotpotQA-tuned extractor while preserving mandatory provenance.",
            "- Add a learned reranker for fused text/vector/graph evidence.",
            "- Add richer entity linking for aliases and redirects.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_failures(run_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Collect failed cases for the report."""
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for record in read_jsonl(run_dir / method / "predictions.jsonl"):
            metrics = record.get("metrics") or evaluate_record(record)
            if metrics.get("em", 0.0) < 1.0:
                record["metrics"] = metrics
                rows.append(record)
    return rows[:limit]


def compare_agent_vs_graph(run_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    """Compare AgenticGraphRAG and GraphRAG-hop2 by question id."""
    graph = {r.get("qid"): r for r in read_jsonl(run_dir / "GraphRAG-hop2" / "predictions.jsonl")}
    agent = {r.get("qid"): r for r in read_jsonl(run_dir / "AgenticGraphRAG" / "predictions.jsonl")}
    rows = []
    for qid in sorted(set(graph) & set(agent)):
        graph_f1 = evaluate_record(graph[qid]).get("f1", 0.0)
        agent_f1 = evaluate_record(agent[qid]).get("f1", 0.0)
        rows.append({"qid": qid, "graph_f1": graph_f1, "agent_f1": agent_f1, "delta": agent_f1 - graph_f1})
    rows.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return rows[:limit]


def _fmt_nullable(value: Any) -> str:
    """Format nullable metric values for Markdown tables."""
    if value is None:
        return "null"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
