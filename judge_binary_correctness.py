#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


RESULT_DIR = Path("data/results")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    for _api_file in (Path("vg_graphrag/tools/api"), Path("vg_graphragVG/tools/api")):
        if _api_file.exists():
            OPENAI_API_KEY = _api_file.read_text(encoding="utf-8").strip()
            if OPENAI_API_KEY:
                break
JUDGE_MODEL = "gpt-5-mini"
MAX_WORKERS = 5
TIMEOUT = 60
MAX_RETRIES = 3
BATCH_SAVE = 5
MAX_TEXT_LEN = 8000

PROMPT = """You are an expert evaluator for agricultural goat-care QA.

Judge whether the model prediction is correct given the question and the ground truth answer.

Question:
{question}

Ground truth answer:
{ground_truth}

Model prediction:
{prediction}

Decision rules:
- Answer "Yes" if the prediction captures the essential meaning, key facts, and main mechanism of the ground truth answer.
- Paraphrases are acceptable.
- The prediction does not need to match the wording of the ground truth exactly.
- Extra information is acceptable only if it does not contradict the ground truth or distract from the answer.
- Answer "No" if the prediction misses a key required point, gives a different primary cause/mechanism/action, is too vague to verify, or contains a contradiction.
- For scenario or indirect QA, answer "Yes" only if the prediction identifies the same main diagnosis, mechanism, management action, or priority as the ground truth.

Return exactly one word:
Yes
or
No

Do not explain."""


def strip_think(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL)
    s = re.sub(r"^```(?:text|json)?\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"\s*```$", "", s.strip())
    return s.strip()


def norm_method(rec: Dict[str, Any]) -> str:
    m = str(rec.get("method") or rec.get("variant") or "").strip()
    if not m:
        hop = rec.get("hop")
        if hop == 2:
            return "GraphRAG-hop2"
        if hop == 0:
            return "VectorRAG"
        if hop == -1:
            return "LLM-only"
    if m in {"hop_2", "hop2", "graph_hop2"}:
        return "GraphRAG-hop2"
    if m in {"hop_0", "vector", "vectorrag"}:
        return "VectorRAG"
    if m in {"hop_-1", "llm_only", "llm"}:
        return "LLM-only"
    return m


def key_of(rec: Dict[str, Any]) -> Tuple[str, str]:
    return norm_method(rec), str(rec.get("question_id"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def resolve_text_fields(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    q = str(rec.get("question") or "")
    gt = str(rec.get("ground_truth") or rec.get("reference_answer") or "")
    pred = str(rec.get("prediction") or rec.get("predicted_answer") or rec.get("final_answer") or "")
    return q, gt, pred


def parse_yes_no(raw: str) -> Tuple[str, bool]:
    s = strip_think(raw)
    token = s.splitlines()[0].strip() if s else ""
    if token == "Yes":
        return "Yes", True
    if token == "No":
        return "No", True
    return token, False


def load_done_progress(progress_path: Path) -> Tuple[set, Dict[Tuple[str, str], Dict[str, Any]]]:
    done: set = set()
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not progress_path.exists():
        return done, rows
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except Exception:
                continue
            k = (str(r.get("method")), str(r.get("question_id")))
            rows[k] = r
            # Only treat strict Yes/No parsed rows as done.
            if bool(r.get("binary_parse_ok")):
                done.add(k)
    return done, rows


def call_one(client: openai.OpenAI, rec: Dict[str, Any]) -> Dict[str, Any]:
    q, gt, pred = resolve_text_fields(rec)
    if len(gt) > MAX_TEXT_LEN:
        gt = gt[:MAX_TEXT_LEN] + "\n...[truncated]"
    if len(pred) > MAX_TEXT_LEN:
        pred = pred[:MAX_TEXT_LEN] + "\n...[truncated]"
    prompt = PROMPT.format(question=q, ground_truth=gt, prediction=pred)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            # gpt-5-mini may consume reasoning tokens first; use responses API with enough output budget.
            r2 = client.responses.create(
                model=JUDGE_MODEL,
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=256,
                reasoning={"effort": "low"},
                timeout=TIMEOUT,
            )
            raw = (getattr(r2, "output_text", "") or "").strip()
            label, ok = parse_yes_no(raw)
            out = {
                "question_id": str(rec.get("question_id")),
                "method": norm_method(rec),
                "binary_correctness_raw": raw,
                "binary_correctness_label": label if ok else None,
                "binary_correct": True if label == "Yes" and ok else False if ok else None,
                "binary_parse_ok": ok,
                "binary_judge_model": JUDGE_MODEL,
            }
            return out
        except Exception as exc:
            last_err = str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
    return {
        "question_id": str(rec.get("question_id")),
        "method": norm_method(rec),
        "binary_correctness_raw": f"[ERROR] {last_err}",
        "binary_correctness_label": None,
        "binary_correct": None,
        "binary_parse_ok": False,
        "binary_judge_model": JUDGE_MODEL,
    }


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    methods = {}
    for m, items in by_method.items():
        parse_fail = sum(1 for x in items if not x.get("binary_parse_ok"))
        yes = sum(1 for x in items if x.get("binary_correct") is True)
        no = sum(1 for x in items if x.get("binary_correct") is False)
        denom = yes + no
        methods[m] = {
            "count": len(items),
            "binary_correct_count": yes,
            "binary_incorrect_count": no,
            "parse_failure_count": parse_fail,
            "binary_accuracy": (yes / denom) if denom > 0 else 0.0,
        }
    return {
        "judge_model": JUDGE_MODEL,
        "record_count": len(rows),
        "method_summary": methods,
    }


def main() -> int:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running judge_binary_correctness.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_file", default="data/results/binary_correctness_judge.jsonl")
    ap.add_argument("--summary_file", default="data/results/binary_correctness_judge.summary.json")
    ap.add_argument("--max_items", type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.input_file)
    if not inp.is_absolute():
        inp = Path(args.input_file)
    if not inp.exists():
        raise FileNotFoundError(inp)
    outp = Path(args.output_file)
    if not outp.is_absolute():
        outp = Path(args.output_file)
    sump = Path(args.summary_file)
    if not sump.is_absolute():
        sump = Path(args.summary_file)

    rows = load_jsonl(inp)
    if args.max_items:
        rows = rows[: args.max_items]

    progress = outp.with_suffix(".progress.jsonl")
    done, done_rows = load_done_progress(progress)

    todo = [r for r in rows if key_of(r) not in done]
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    lock = threading.Lock()
    buf: List[Dict[str, Any]] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        progress.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            with progress.open("a", encoding="utf-8") as f:
                for x in buf:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
        buf = []

    total = len(rows)
    completed = len(done)
    print(f"Binary judge input={len(rows)} done={len(done)} remaining={len(todo)}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(call_one, client, r): r for r in todo}
        for fut in as_completed(futs):
            res = fut.result()
            buf.append(res)
            completed += 1
            if len(buf) >= BATCH_SAVE:
                flush()
            if completed % 10 == 0 or completed == total:
                print(f"  [{completed}/{total}] {res.get('method')} Q{res.get('question_id')} -> {res.get('binary_correctness_label')}")
    flush()

    merged = dict(done_rows)
    for r in load_jsonl(progress):
        merged[(str(r.get("method")), str(r.get("question_id")))] = r
    final_rows: List[Dict[str, Any]] = []
    for r in rows:
        k = key_of(r)
        final_rows.append(merged.get(k, {
            "question_id": str(r.get("question_id")),
            "method": norm_method(r),
            "binary_correctness_raw": "[MISSING]",
            "binary_correctness_label": None,
            "binary_correct": None,
            "binary_parse_ok": False,
            "binary_judge_model": JUDGE_MODEL,
        }))

    save_jsonl(outp, final_rows)
    summary = summarize(final_rows)
    sump.parent.mkdir(parents=True, exist_ok=True)
    sump.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {outp}")
    print(f"Wrote {sump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
