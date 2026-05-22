#!/usr/bin/env python3
"""
judge_openai_151.py
===================
OpenAI API-based LLM Judge for 151-question full ablation results.

Evaluates both prompting and agentic JSONL result files.
Uses gpt-5-mini (OpenAI) as judge — parallel async calls for speed.

Usage:
  python judge_openai_151.py                     # auto-detect latest files
  python judge_openai_151.py --input_file FILE   # specific file
  python judge_openai_151.py --max_items 10      # smoke test
"""
import sys
import io

# ── Force UTF-8 output on Windows ────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import re
import json
import time
import argparse
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

os.chdir(Path(__file__).parent)

import openai

# ── Config ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    for _api_file in (Path("vg_graphrag/tools/api"), Path("vg_graphragVG/tools/api")):
        if _api_file.exists():
            OPENAI_API_KEY = _api_file.read_text(encoding="utf-8").strip()
            if OPENAI_API_KEY:
                break
JUDGE_MODEL   = "gpt-5-mini"  # OpenAI judge model (e.g., "gpt-5-mini", "gpt-5.1", etc.)
RESULT_DIR    = Path("data/results")
BATCH_SAVE    = 5           # save every N items (more frequent checkpointing)
MAX_WORKERS   = 5           # increase concurrency for faster evaluation
TIMEOUT       = 60          # seconds per call
MAX_RETRIES   = 3
MAX_PREDICTED_LEN = 8000    # truncate predicted_answer beyond this (prevent huge prompts)
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_SEED = 42

QA_EVAL_PROMPT = """\

You are an expert evaluator for agricultural goat-care QA systems.
Evaluate the **Predicted Answer** against the **Reference Answer** for the given question.

**Question:**
{question}
**Reference Answer (Gold Standard):**
{reference_answer}
**Predicted Answer (Model Output):**
{predicted_answer}

**Evaluation Criteria (each 1–5):**

1. **Coverage**: How completely does the predicted answer cover all the critical points, steps, and nuances present in the reference answer? (Punish missing information).
2. **Factual Correctness**: Are all the facts, data, and claims in the predicted answer factually accurate and logically consistent with the reference? (Punish factual contradictions).
3. **Topic Adherence**: Does the predicted answer stay strictly focused on addressing the user's specific Question? (Punish rambling, verbosity, or bringing in distantly related tangential graph noise).

**Output format – respond with ONLY valid JSON (no markdown code blocks, no extra text):**
{{
  "per_criterion": {{
    "coverage":            <int 1-5>,
    "factual_correctness": <int 1-5>,
    "topic_adherence":     <int 1-5>
  }},
  "ascore": <float 1.0-5.0>,
  "llm_f1": <float 0.0-1.0>,
  "differences": "<briefly list missing coverage, factual errors, or tangential rambling>",
  "assessment": "<one-sentence overall verdict>",
  "overall_rating": <int 1-5>
}}
`llm_f1` = fraction of reference key-points explicitly covered by the predicted answer (0.0=none, 1.0=all), based on your semantic judgment.
`ascore` = average of coverage, factual_correctness, and topic_adherence.
"""

# ── Helpers ────────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()


def call_judge(client: openai.OpenAI, record: dict) -> dict:
    """Call OpenAI judge for one record. Returns merged record with scores."""
    question  = record.get("question", "")
    reference = record.get("reference_answer", "")
    predicted = record.get("predicted_answer", "") or ""
    # Truncate excessively long answers to prevent context window overflow
    if len(predicted) > MAX_PREDICTED_LEN:
        predicted = predicted[:MAX_PREDICTED_LEN] + "\n...[truncated]"

    prompt = QA_EVAL_PROMPT.format(
        question=question,
        reference_answer=reference,
        predicted_answer=predicted,
    )

    for attempt in range(MAX_RETRIES):
        try:
            request_kwargs = dict(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000,
                timeout=TIMEOUT,
                temperature=JUDGE_TEMPERATURE,
                top_p=JUDGE_TOP_P,
                seed=JUDGE_SEED,
            )
            response = None
            for _compat_try in range(4):
                try:
                    response = client.chat.completions.create(**request_kwargs)
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    removed = False
                    # Some models only support default temperature/top_p values.
                    if "temperature" in msg and "unsupported" in msg and "temperature" in request_kwargs:
                        request_kwargs.pop("temperature", None)
                        removed = True
                    if "top_p" in msg and "unsupported" in msg and "top_p" in request_kwargs:
                        request_kwargs.pop("top_p", None)
                        removed = True
                    if "seed" in msg and "unsupported" in msg and "seed" in request_kwargs:
                        request_kwargs.pop("seed", None)
                        removed = True
                    if not removed:
                        raise
            if response is None:
                raise RuntimeError("Judge call failed after compatibility fallbacks")
            raw = response.choices[0].message.content or ""
            raw = strip_think(raw).strip()

            # Strip markdown code blocks if present
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
            raw = raw.strip()

            scores = json.loads(raw)

            # Flatten per_criterion into top-level
            for crit, val in scores.get("per_criterion", {}).items():
                scores[crit] = val
            scores.pop("per_criterion", None)
            vals = []
            for key in ("coverage", "factual_correctness", "topic_adherence"):
                try:
                    vals.append(float(scores.get(key)))
                except (TypeError, ValueError):
                    pass
            scores["ascore"] = sum(vals) / len(vals) if vals else None

            # Merge into record
            out = dict(record)
            out.update(scores)
            return out

        except json.JSONDecodeError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            out = dict(record)
            out.update({
                "overall_rating": None, "llm_f1": None,
                "coverage": None, "factual_correctness": None,
                "topic_adherence": None,
                "differences": f"[Error: {e}]",
                "assessment": "[Error]",
            })
            return out

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            out = dict(record)
            out.update({
                "overall_rating": None, "llm_f1": None,
                "coverage": None, "factual_correctness": None,
                "topic_adherence": None,
                "differences": f"[APIError: {e}]",
                "assessment": "[APIError]",
            })
            return out


_lock = threading.Lock()

def load_progress(progress_path: Path) -> set:
    """Load already-completed records. Only treats records with valid scores as done."""
    if not progress_path.exists():
        return set()
    done = set()
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    # Only consider a record done if it has a valid overall_rating
                    if r.get("overall_rating") is not None:
                        done.add((r.get("variant"), str(r.get("question_id"))))
                except Exception:
                    pass
    return done


def evaluate_file(input_path: Path, output_path: Path, max_items: int = None):
    """Evaluate all records in a JSONL file using OpenAI judge."""
    print(f"\n{'='*70}")
    print(f"Input:  {input_path.name}")
    print(f"Output: {output_path.name}")
    print(f"Judge:  {JUDGE_MODEL}")
    print(f"{'='*70}")

    # Load records
    records = []
    for line in input_path.read_text(encoding="utf-8").strip().splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            pass

    # Normalize keys for datasets that do not include variant / question_id fields.
    # Include combo folder name to avoid cross-file key collisions.
    combo_name = input_path.parent.name
    default_variant = f"{combo_name}_{input_path.stem}"
    for idx, r in enumerate(records):
        if not r.get("variant"):
            r["variant"] = default_variant
        if r.get("question_id") is None:
            r["question_id"] = idx

    if max_items:
        records = records[:max_items]
    print(f"Records to judge: {len(records)}")

    # Resume from progress
    progress_path = output_path.with_suffix("").with_suffix(".progress.jsonl")
    done_keys = load_progress(progress_path)
    remaining = [r for r in records
                 if (r.get("variant"), str(r.get("question_id"))) not in done_keys]
    print(f"Already done: {len(records)-len(remaining)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All done! Merging progress file...")
        _merge_to_output(progress_path, output_path, records)
        return

    # Init OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Work concurrently
    judged = list(done_keys)  # we'll track counts
    completed = len(done_keys)
    total = len(records)
    batch_buf = []

    def save_batch():
        nonlocal batch_buf
        if batch_buf:
            with _lock:
                with open(progress_path, "a", encoding="utf-8") as pf:
                    for item in batch_buf:
                        pf.write(json.dumps(item, ensure_ascii=False) + "\n")
            batch_buf = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(call_judge, client, r): r for r in remaining}
        for future in as_completed(futures):
            result = future.result()
            with _lock:
                batch_buf.append(result)
                completed += 1

            if len(batch_buf) >= BATCH_SAVE:
                save_batch()

            if completed % 10 == 0 or completed == total:
                v = result.get("variant", "?")
                qid = result.get("question_id", "?")
                ovr = result.get("overall_rating", "?")
                f1  = result.get("llm_f1", "?")
                print(f"  [{completed}/{total}] {v} Q{qid} → Overall={ovr} F1={f1}")

    # Save any remaining batch
    save_batch()

    # Merge to final output
    _merge_to_output(progress_path, output_path, records)


def _merge_to_output(progress_path: Path, output_path: Path, original_records: list):
    """Merge progress JSONL with original ordering into final output file."""
    # Load all judged records
    judged_map = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").strip().splitlines():
            try:
                r = json.loads(line)
                judged_map[(r.get("variant"), str(r.get("question_id")))] = r
            except Exception:
                pass

    out_records = []
    for r in original_records:
        key = (r.get("variant"), str(r.get("question_id")))
        if key in judged_map:
            out_records.append(judged_map[key])
        else:
            out_records.append(r)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Merged {len(out_records)} records → {output_path}")
    _print_summary(out_records)


def _print_summary(records: list):
    """Print per-variant summary."""
    from collections import defaultdict
    variant_data = defaultdict(list)
    for r in records:
        ovr  = r.get("overall_rating")
        asc  = r.get("ascore")
        f1   = r.get("llm_f1")
        cov  = r.get("coverage")
        fct  = r.get("factual_correctness")
        tad  = r.get("topic_adherence")
        if ovr is not None:
            if asc is None and cov is not None and fct is not None and tad is not None:
                asc = (float(cov) + float(fct) + float(tad)) / 3
            variant_data[r["variant"]].append((ovr, asc or 0, f1 or 0, cov or 0, fct or 0, tad or 0))

    print(f"\n{'─'*110}")
    print(f"{'Variant':<30}  {'n':>4}  {'Overall':>7}  {'ASCORE':>7}  {'LLM-F1':>7}  {'Coverag':>7}  {'FactCor':>7}  {'TopAdhr':>7}")
    print(f"{'─'*110}")
    for v, vals in sorted(variant_data.items()):
        n   = len(vals)
        ovr = sum(x[0] for x in vals) / n
        asc = sum(x[1] for x in vals) / n
        f1  = sum(x[2] for x in vals) / n
        cov = sum(x[3] for x in vals) / n
        fct = sum(x[4] for x in vals) / n
        tad = sum(x[5] for x in vals) / n
        print(f"{v:<30}  {n:>4}  {ovr:>7.3f}  {asc:>7.3f}  {f1:>7.4f}  {cov:>7.3f}  {fct:>7.3f}  {tad:>7.3f}")
    print(f"{'─'*110}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running judge_openai.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default=None,
                        help="Specific JSONL input (auto-detects if omitted)")
    parser.add_argument("--output_file", default=None,
                        help="Optional judged JSONL output path")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--mode", choices=["prompting", "agentic", "qaset1", "all"], default="all")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.input_file:
        inp = Path(args.input_file)
        if not inp.is_absolute() and not inp.exists():
            inp = RESULT_DIR / inp
        # Keep a stable but unique output filename per combo/stage for true resume.
        combo_name = inp.parent.name
        out = Path(args.output_file) if args.output_file else RESULT_DIR / f"judge_openai_{combo_name}_{inp.stem}.jsonl"
        if not out.is_absolute() and out.parent == Path("."):
            out = RESULT_DIR / out
        evaluate_file(inp, out, args.max_items)
        return

    # Auto-detect latest files
    files_to_judge = []

    if args.mode in ("qaset1", "all"):
        # Accept both old qaset1_multimodel_* and new graphrag_qaset1_* files
        qaset1_files = sorted(
            [f for f in RESULT_DIR.glob("qaset1_multimodel_*.jsonl") if "progress" not in f.name]
            + [f for f in RESULT_DIR.glob("graphrag_qaset1_*.jsonl")   if "progress" not in f.name]
        )
        if qaset1_files:
            files_to_judge.append(("qaset1", qaset1_files[-1]))
        elif args.mode == "qaset1":
            print("⚠ No qaset1_multimodel_*.jsonl or graphrag_qaset1_*.jsonl file found.")

    if args.mode in ("prompting", "all"):
        prompting = sorted(RESULT_DIR.glob("prompting_ablation_qa_151_*.jsonl"))
        if not prompting:
            prompting = sorted(RESULT_DIR.glob("prompting_ablation_qa_2*.jsonl"))
        if prompting:
            files_to_judge.append(("prompting", prompting[-1]))
        elif args.mode == "prompting":
            print("⚠ No prompting ablation file found.")

    if args.mode in ("agentic", "all"):
        agentic = sorted(RESULT_DIR.glob("agentic_ablation_qa_151_*.jsonl"))
        if not agentic:
            agentic = sorted(RESULT_DIR.glob("agentic_ablation_qa_2*.jsonl"))
            agentic = [f for f in agentic if "corrupt" not in f.name]
        if agentic:
            files_to_judge.append(("agentic", agentic[-1]))
        elif args.mode == "agentic":
            print("⚠ No agentic ablation file found.")

    if not files_to_judge:
        print("ERROR: No files to judge."); sys.exit(1)

    for label, inp_path in files_to_judge:
        out_path = RESULT_DIR / f"judge_openai_{label}_{ts}.jsonl"
        evaluate_file(inp_path, out_path, args.max_items)

    print("\n✅ All judge runs complete.")


if __name__ == "__main__":
    main()
