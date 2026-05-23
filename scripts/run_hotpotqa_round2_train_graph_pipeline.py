#!/usr/bin/env python
"""Run the HotpotQA train-graph / validation-eval Round 2 pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], trace: Path) -> None:
    """Run a command and append its output to trace."""
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("a", encoding="utf-8") as f:
        f.write(str({"cmd": cmd, "returncode": proc.returncode, "elapsed_sec": time.perf_counter() - started, "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-3000:]}) + "\n")
    print(f"[{proc.returncode}] {' '.join(cmd)}")
    if proc.stdout.strip():
        print(proc.stdout[-1200:])
    if proc.stderr.strip():
        print(proc.stderr[-1200:])
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> int:
    """Execute graph build/reuse, validation prep, evaluation, reports, and compileall."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-run-id", default="hotpotqa_train_full_neo4j_v1")
    parser.add_argument("--run-name", default="hotpotqa_round2_train_graph_validation")
    parser.add_argument("--eval-size", default="full")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-train-docs", type=int, default=None, help="Debug only; omit for full train graph.")
    args = parser.parse_args()
    py = sys.executable
    trace = ROOT / "runs" / args.run_name / "pipeline_trace.jsonl"
    build_cmd = [py, "scripts/build_hotpotqa_train_neo4j_graph.py", "--graph-run-id", args.graph_run_id]
    if args.force_rebuild:
        build_cmd.append("--force-rebuild")
    if args.max_train_docs is not None:
        build_cmd.extend(["--max-docs", str(args.max_train_docs)])
    run([py, "scripts/check_env.py"], trace)
    run(build_cmd, trace)
    run([py, "scripts/prepare_hotpotqa.py", "--setting", "distractor", "--split", "validation", "--sample-size", "full", "--output-suffix", "validation_full"], trace)
    run([py, "scripts/run_hotpotqa_train_graph_experiment.py", "--all", "--run-name", args.run_name, "--graph-run-id", args.graph_run_id, "--eval-size", args.eval_size], trace)
    run([py, "scripts/write_hotpotqa_round2_bilingual_report.py", "--run-dir", f"runs/{args.run_name}", "--graph-manifest", f"data/graph_runs/{args.graph_run_id}/run_manifest.json"], trace)
    run([py, "-m", "compileall", "."], trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
