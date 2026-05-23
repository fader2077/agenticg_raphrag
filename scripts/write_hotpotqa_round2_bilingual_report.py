#!/usr/bin/env python
"""Write Chinese and English HotpotQA Round 2 reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import METHODS, read_json, read_jsonl


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    return float(value or 0)


def _escape(text: Any, limit: int = 120) -> str:
    value = str(text or "").replace("\n", " ").replace("|", "\\|")
    return value[:limit] + ("..." if len(value) > limit else "")


def _metrics_table(all_metrics: dict[str, dict[str, Any]]) -> list[str]:
    """Return a compact Markdown metrics table."""
    lines = [
        "| Method | N | EM | F1 | SF P | SF R | SF F1 | Gold Title R | Path Found | Path Recall | Tool Calls | Repair | Judge Bin | OpenAI Judge | Judge Err | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        m = all_metrics.get(method, {})
        if not m:
            continue
        lines.append(
            f"| {method} | {int(m.get('count', 0))} | {_metric(m, 'em'):.4f} | {_metric(m, 'f1'):.4f} | "
            f"{_metric(m, 'supporting_fact_precision'):.4f} | {_metric(m, 'supporting_fact_recall'):.4f} | "
            f"{_metric(m, 'supporting_fact_f1'):.4f} | {_metric(m, 'gold_title_recall'):.4f} | "
            f"{_metric(m, 'path_found_rate'):.4f} | {_metric(m, 'path_recall'):.4f} | "
            f"{_metric(m, 'avg_tool_calls'):.2f} | {_metric(m, 'repair_used'):.4f} | "
            f"{_metric(m, 'judge_binary_accuracy'):.4f} | {_metric(m, 'openai_judge_score_mean'):.4f} | "
            f"{_metric(m, 'judge_error_rate'):.4f} | {_metric(m, 'avg_latency_ms'):.1f} |"
        )
    return lines


def _failure_cases(run_dir: Path, method: str = "AgenticGraphRAG", limit: int = 10) -> list[str]:
    """Return representative failed rows for a method."""
    rows = []
    for row in read_jsonl(run_dir / method / "predictions.jsonl"):
        if (row.get("metrics") or {}).get("em", 0) < 1:
            rows.append(row)
        if len(rows) >= limit:
            break
    if not rows:
        return ["- No failed cases found."]
    return [
        f"{i}. `{r.get('qid')}` Q={_escape(r.get('question'))} pred=`{_escape(r.get('pred_answer'), 80)}` gold=`{_escape(r.get('gold_answer'), 80)}`"
        for i, r in enumerate(rows, start=1)
    ]


def _agent_vs_graph_cases(run_dir: Path, limit: int = 10) -> list[str]:
    """List AgenticGraphRAG improvements or regressions versus GraphRAG-hop2."""
    graph_rows = {r.get("qid"): r for r in read_jsonl(run_dir / "GraphRAG-hop2" / "predictions.jsonl")}
    agent_rows = {r.get("qid"): r for r in read_jsonl(run_dir / "AgenticGraphRAG" / "predictions.jsonl")}
    diffs = []
    for qid, agent in agent_rows.items():
        graph = graph_rows.get(qid)
        if not graph:
            continue
        a_f1 = float((agent.get("metrics") or {}).get("f1", 0) or 0)
        g_f1 = float((graph.get("metrics") or {}).get("f1", 0) or 0)
        delta = a_f1 - g_f1
        if abs(delta) > 1e-9:
            diffs.append((delta, qid, graph, agent))
    diffs.sort(key=lambda item: abs(item[0]), reverse=True)
    if not diffs:
        return ["- No F1-different AgenticGraphRAG vs GraphRAG-hop2 cases found."]
    out = []
    for i, (delta, qid, graph, agent) in enumerate(diffs[:limit], start=1):
        label = "improved" if delta > 0 else "regressed"
        out.append(
            f"{i}. `{qid}` {label} delta_f1={delta:.3f}; "
            f"Graph=`{_escape(graph.get('pred_answer'), 60)}` Agent=`{_escape(agent.get('pred_answer'), 60)}` Gold=`{_escape(agent.get('gold_answer'), 60)}`"
        )
    return out


def _error_categories(run_dir: Path) -> dict[str, int]:
    """Build a simple error category count from prediction traces and error logs."""
    counts = Counter(
        {
            "retrieval_miss": 0,
            "partial_evidence": 0,
            "wrong_entity_link": 0,
            "graph_no_path": 0,
            "noisy_graph": 0,
            "unsupported_answer": 0,
            "judge_error": 0,
            "generation_error": 0,
            "over_repair": 0,
        }
    )
    for method in METHODS:
        pred_path = run_dir / method / "predictions.jsonl"
        if pred_path.exists():
            for row in read_jsonl(pred_path):
                metrics = row.get("metrics") or {}
                if method != "LLM-only" and not row.get("retrieved_chunks"):
                    counts["retrieval_miss"] += 1
                sf_recall = float(metrics.get("supporting_fact_recall", 0) or 0)
                if 0 < sf_recall < 1:
                    counts["partial_evidence"] += 1
                if method.startswith("GraphRAG") and not row.get("retrieved_paths"):
                    counts["graph_no_path"] += 1
                if (row.get("verifier") or {}).get("verdict") in {"unsupported", "partially_supported", "insufficient_evidence"}:
                    counts["unsupported_answer"] += 1
                if (row.get("judge") or {}).get("judge_error"):
                    counts["judge_error"] += 1
                if method == "AgenticGraphRAG" and float(metrics.get("repair_used", 0) or 0) > 0 and float(metrics.get("repair_success", 0) or 0) == 0:
                    counts["over_repair"] += 1
        err_path = run_dir / method / "errors.jsonl"
        if err_path.exists():
            for err in read_jsonl(err_path):
                etype = str(err.get("error_type") or "")
                stage = str(err.get("stage") or "")
                if "graph_no_path" in etype:
                    counts["graph_no_path"] += 1
                if "judge" in stage or "judge" in etype:
                    counts["judge_error"] += 1
                if "generation" in stage:
                    counts["generation_error"] += 1
    return dict(counts)


def _category_table(categories: dict[str, int]) -> list[str]:
    lines = ["| Category | Count |", "|---|---:|"]
    for key, value in categories.items():
        lines.append(f"| {key} | {value} |")
    return lines


def _shared_sections(run_dir: Path, graph_manifest: dict[str, Any], manifest: dict[str, Any], all_metrics: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    table = _metrics_table(all_metrics)
    categories = _category_table(_error_categories(run_dir))
    failures = _failure_cases(run_dir)
    comparisons = _agent_vs_graph_cases(run_dir)
    counts = json.dumps(graph_manifest.get("counts", {}), ensure_ascii=False)
    return {
        "table": table,
        "categories": categories,
        "failures": failures,
        "comparisons": comparisons,
        "graph": [
            f"- graph_run_id: `{graph_manifest.get('graph_run_id')}`",
            f"- status: `{graph_manifest.get('status')}`",
            f"- Neo4j URI: `{graph_manifest.get('neo4j_uri')}`",
            f"- chunk fulltext index: `{graph_manifest.get('fulltext_index_name')}`",
            f"- counts: `{counts}`",
            f"- reuse policy: {graph_manifest.get('reuse_policy')}",
            f"- run methods: `{', '.join(manifest.get('methods', []))}`",
            f"- eval question count: `{manifest.get('eval_question_count')}`",
        ],
    }


def main() -> int:
    """Generate bilingual reports from a run directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--graph-manifest", default="data/graph_runs/hotpotqa_train_full_neo4j_v1/run_manifest.json")
    parser.add_argument("--zh-report", default="reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_ZH.md")
    parser.add_argument("--en-report", default="reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_EN.md")
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    all_metrics = read_json(run_dir / "all_metrics.json", {}) or {}
    manifest = read_json(run_dir / "experiment_manifest.json", {}) or {}
    graph_manifest = read_json(ROOT / args.graph_manifest, {}) or {}
    sections = _shared_sections(run_dir, graph_manifest, manifest, all_metrics)

    zh = [
        "# HotpotQA Round 2 訓練集建圖 / 驗證集評估報告",
        "",
        "## 結論",
        "- 本輪改成使用 HotpotQA distractor `train` split 建立 Neo4j 圖譜，使用 `validation` split 做正式評估。",
        "- Hugging Face `hotpot_qa/distractor` 沒有帶答案的公開 `test` split；可評估 split 是 validation/dev。",
        "- 圖譜狀態保存在同一個 Neo4j `graph_run_id`，所有方法與消融共用，不會每次重建圖譜。",
        "- 已使用 `neo4j_graphrag.indexes.create_fulltext_index` 建立 chunk fulltext index，也建立 entity fulltext index 供 entity linking 使用。",
        "",
        "## 資料與輸出位置",
        "- 原始資料來源：Hugging Face `hotpot_qa/distractor`，loader 會優先使用本地 cache，缺失時才下載。",
        "- 評估資料：`data/processed/hotpotqa_validation_full`，共 7,405 題。",
        "- 正式 run：`runs/hotpotqa_round2_train_graph_validation`。",
        "- 中文報告：`reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_ZH.md`。",
        "- English report：`reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_EN.md`。",
        "",
        "## Neo4j 圖譜狀態",
        *sections["graph"],
        "",
        "## 模型確認",
        "- `config.py` 中確認的既有 Ollama 設定：`qa_model=qwen2.5:7b-instruct-fp16`、`graph_create_model=deepseekr1-14b-fp16`、`embed_model=nomic-embed-text:latest`。",
        "- 本輪 full-train 建圖沒有呼叫 LLM triple extractor；實際建圖模型記錄為 `deterministic_titlecase_cooccurrence_v2`，原因是 train split 規模很大，LLM 抽取成本與時間不可控。",
        "- QA generation 使用可重現的 `deterministic_evidence_sentence_v1` fallback，目標是驗證 pipeline、trace、retrieval、judge 與 metrics，而不是宣稱最佳答案品質。",
        "- VectorRAG 在本輪 full train graph 中明確記錄為 `neo4j_fulltext_fallback_no_full_train_embeddings`；未建立 818k chunks 的 dense embedding index。",
        "",
        "## AgenticGraphRAG 實作狀態",
        "- 已實作 bounded deterministic controller：analyze_question、text_search、vector_search、entity_link、graph_expand、path_rank、evidence_fusion、answer_generate、evidence_verify、repair_retrieve。",
        "- Agent trace 已寫入每題 `tool_trace`；full run 平均 tool calls 見 metrics table。",
        "- 已檢查 `vg_graphragVG/pipeline`、`tools`、`adapters`、`demo`；因部分 VG 原始 `.py` 缺失但 `.pyc` 仍在，新增 `vg_graphrag` import shim 與 generic compatibility modules，使既有專案可 import。",
        "- 正式 HotpotQA 評估使用 HotpotQA 專用 controller，避免 goat domain policy 汙染主實驗。",
        "",
        "## Metrics",
        *sections["table"],
        "",
        "## Judge 統計",
        "- `judge_binary_correctness.py` 與 `judge_openai.py` 已透過 `src/eval/judge_adapter.py` 統一整合。",
        "- API key 已可由 `C:\\Users\\kbllm\\Downloads\\api.txt` 讀取並設定到環境，但報告不顯示 key。",
        "- 本輪未設定 `HOTPOTQA_RUN_OPENAI_JUDGE_CALLS=1`，因此正式 judge 欄位使用本地 deterministic fallback；`judge_error_rate` 為 0。",
        "",
        "## 錯誤分類",
        *sections["categories"],
        "",
        "## 失敗案例（AgenticGraphRAG）",
        *sections["failures"],
        "",
        "## AgenticGraphRAG vs GraphRAG-hop2 差異案例",
        *sections["comparisons"],
        "",
        "## 限制",
        "- 嚴格 train-only graph 不保證包含 validation gold supporting pages；這會降低 retrieval recall，但符合 split separation。",
        "- Graph relation extraction 是 deterministic titlecase co-occurrence，不等同 LLM relation extraction。",
        "- OpenAI 外部 judge 已整合但本輪未實際呼叫，避免大量 API 成本；可用環境變數開啟。",
        "",
        "## 下一步",
        "- 對 train chunks 建可重用 dense vector index，讓 VectorRAG 與 Agent vector route 不再使用 fulltext fallback。",
        "- 加入可批次化的 Ollama/LLM relation extraction ablation，只在 train graph 建置階段執行一次。",
        "- 對 super-node 做 degree cap 或 relation type filtering，提升 hop2 path precision。",
    ]

    en = [
        "# HotpotQA Round 2 Train-Graph / Validation Evaluation Report",
        "",
        "## Conclusion",
        "- This round builds the Neo4j graph from the HotpotQA distractor `train` split and evaluates on `validation`.",
        "- Hugging Face `hotpot_qa/distractor` has no answer-labeled public `test` split; validation/dev is the evaluable split here.",
        "- The graph state is saved under one Neo4j `graph_run_id`; all methods and ablations reuse it without rebuilding.",
        "- `neo4j_graphrag.indexes.create_fulltext_index` was used for the chunk fulltext index, and an entity fulltext index was added for entity linking.",
        "",
        "## Data And Artifacts",
        "- Raw data source: Hugging Face `hotpot_qa/distractor`; the loader prefers the local cache and downloads only when missing.",
        "- Evaluation data: `data/processed/hotpotqa_validation_full`, 7,405 questions.",
        "- Full run: `runs/hotpotqa_round2_train_graph_validation`.",
        "- Chinese report: `reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_ZH.md`.",
        "- English report: `reports/HOTPOTQA_ROUND2_TRAIN_GRAPH_REPORT_EN.md`.",
        "",
        "## Neo4j Graph State",
        *sections["graph"],
        "",
        "## Model Accounting",
        "- Existing `config.py` Ollama settings are `qa_model=qwen2.5:7b-instruct-fp16`, `graph_create_model=deepseekr1-14b-fp16`, and `embed_model=nomic-embed-text:latest`.",
        "- The full-train graph build did not call an LLM triple extractor; the recorded graph model is `deterministic_titlecase_cooccurrence_v2` because full train is too large for uncontrolled LLM extraction cost.",
        "- QA generation uses the reproducible `deterministic_evidence_sentence_v1` fallback; the goal is traceable pipeline validation, not best possible answer quality.",
        "- VectorRAG is explicitly recorded as `neo4j_fulltext_fallback_no_full_train_embeddings`; no dense vector index was built over 818k chunks.",
        "",
        "## AgenticGraphRAG Implementation",
        "- The bounded deterministic controller implements analyze_question, text_search, vector_search, entity_link, graph_expand, path_rank, evidence_fusion, answer_generate, evidence_verify, and repair_retrieve.",
        "- Per-question `tool_trace` is written for AgenticGraphRAG; average tool calls are shown in the metrics table.",
        "- `vg_graphragVG/pipeline`, `tools`, `adapters`, and `demo` were inspected. Some VG `.py` files were missing while `.pyc` files remained, so a `vg_graphrag` import shim and generic compatibility modules were added.",
        "- The formal HotpotQA run uses a HotpotQA-specific controller to avoid goat-domain policy contamination.",
        "",
        "## Metrics",
        *sections["table"],
        "",
        "## Judge Statistics",
        "- `judge_binary_correctness.py` and `judge_openai.py` are unified through `src/eval/judge_adapter.py`.",
        "- The API key can be loaded from `C:\\Users\\kbllm\\Downloads\\api.txt` and set in the environment; the key is not printed in reports.",
        "- `HOTPOTQA_RUN_OPENAI_JUDGE_CALLS=1` was not set, so this run used deterministic local judge fallback fields; `judge_error_rate` is 0.",
        "",
        "## Error Categories",
        *sections["categories"],
        "",
        "## Failure Cases (AgenticGraphRAG)",
        *sections["failures"],
        "",
        "## AgenticGraphRAG vs GraphRAG-hop2 Difference Cases",
        *sections["comparisons"],
        "",
        "## Limitations",
        "- A strict train-only graph does not guarantee that validation gold supporting pages are present; this lowers retrieval recall but preserves split separation.",
        "- Graph relation extraction is deterministic titlecase co-occurrence, not LLM relation extraction.",
        "- External OpenAI judge calls are integrated but were not executed in this run to avoid large API cost; they can be enabled by environment variable.",
        "",
        "## Next Steps",
        "- Build a reusable dense vector index for train chunks so VectorRAG and Agent vector routes no longer use fulltext fallback.",
        "- Add a batched Ollama/LLM relation-extraction ablation that runs once during train graph construction.",
        "- Add super-node degree caps or relation-type filters to improve hop2 path precision.",
    ]

    zh_path = ROOT / args.zh_report
    en_path = ROOT / args.en_report
    zh_path.parent.mkdir(parents=True, exist_ok=True)
    en_path.parent.mkdir(parents=True, exist_ok=True)
    zh_path.write_text("\n".join(zh) + "\n", encoding="utf-8")
    en_path.write_text("\n".join(en) + "\n", encoding="utf-8")
    print(f"Wrote {zh_path} and {en_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
