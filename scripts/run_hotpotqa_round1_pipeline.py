#!/usr/bin/env python
"""Two-stage HotpotQA Round 1 pipeline: smoke gate, then full benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import METHODS, load_yaml, read_json, read_jsonl, run_dir, write_json


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a trace file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage(cmd: list[str], trace_path: Path, env: dict[str, str] | None = None, check: bool = True) -> dict[str, Any]:
    """Run one command, capture output, and append it to pipeline trace."""
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    row = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": time.perf_counter() - started,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    append_jsonl(trace_path, row)
    print(f"[{proc.returncode}] {' '.join(cmd)}")
    if proc.stdout.strip():
        print(proc.stdout[-1200:])
    if proc.stderr.strip():
        print(proc.stderr[-1200:])
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return row


def update_manifest(run_path: Path, payload: dict[str, Any]) -> None:
    """Merge payload into a run manifest."""
    manifest = read_json(run_path / "experiment_manifest.json", {}) or {}
    manifest.update(payload)
    write_json(run_path / "experiment_manifest.json", manifest)


def prepare_build_run_eval(
    config_path: str,
    suffix: str,
    sample_size: int,
    report_path: str,
    trace_path: Path,
    env: dict[str, str],
) -> None:
    """Run prepare, build, all methods, and evaluation for one suffix."""
    py = sys.executable
    run_stage([py, "scripts/check_env.py", "--auto-install"], trace_path, env=env)
    run_stage([py, "scripts/prepare_hotpotqa.py", "--config", config_path, "--setting", "distractor", "--sample-size", str(sample_size), "--output-suffix", suffix], trace_path, env=env)
    run_stage([py, "scripts/build_hotpotqa_indexes.py", "--config", config_path, "--sample-suffix", suffix], trace_path, env=env)
    for method in METHODS:
        run_stage([py, "scripts/run_hotpotqa_experiment.py", "--method", method, "--config", config_path, "--sample-suffix", suffix], trace_path, env=env)
    run_name = f"runs/hotpotqa_round1_{suffix}"
    run_stage([py, "scripts/evaluate_hotpotqa_results.py", "--run-dir", run_name, "--write-report", report_path], trace_path, env=env)


def smoke_gate(config: dict[str, Any], run_path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    """Return smoke pass/fail, reasons, and warnings."""
    failures: list[str] = []
    warnings: dict[str, Any] = {}
    all_metrics = read_json(run_path / "all_metrics.json", {}) or {}
    if len(all_metrics) < len(METHODS):
        failures.append(f"Expected {len(METHODS)} metrics entries, found {len(all_metrics)}")
    for method in METHODS:
        pred_path = run_path / method / "predictions.jsonl"
        metrics_path = run_path / method / "metrics.json"
        if not pred_path.exists():
            failures.append(f"{method} missing predictions.jsonl")
            continue
        rows = read_jsonl(pred_path)
        if not rows:
            failures.append(f"{method} has zero predictions")
        if not metrics_path.exists():
            failures.append(f"{method} missing metrics.json")
        if method == "AgenticGraphRAG" and not any(row.get("tool_trace") for row in rows):
            failures.append("AgenticGraphRAG produced no tool_trace")
        if method in {"GraphRAG-hop1", "GraphRAG-hop2"}:
            path_found = float((all_metrics.get(method) or {}).get("path_found", 0.0) or 0.0)
            warnings[f"{method}_path_found_rate"] = path_found
            if path_found == 0.0:
                warnings[f"{method}_warning"] = "graph path found rate is zero"
    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "src", "scripts", "tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    warnings["compileall_returncode"] = compile_result.returncode
    if compile_result.returncode != 0:
        failures.append("compileall failed during smoke gate")
        warnings["compileall_stderr_tail"] = compile_result.stderr[-4000:]
    gate = {"passed": not failures, "failures": failures, "warnings": warnings}
    write_json(run_path / "smoke_gate.json", gate)
    update_manifest(run_path, {"smoke_gate": gate, "smoke_passed": not failures})
    return not failures, failures, warnings


def write_failure_report(path: Path, failures: list[str], warnings: dict[str, Any]) -> None:
    """Write a smoke failure report."""
    lines = [
        "# HotpotQA Round 1 Smoke Failure",
        "",
        "Smoke test did not satisfy the automated gate after retries.",
        "",
        "## Failures",
    ]
    lines.extend(f"- {failure}" for failure in failures)
    lines.extend(["", "## Warnings", "```json", json.dumps(warnings, ensure_ascii=False, indent=2), "```"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the two-stage benchmark pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa_round1.yaml")
    parser.add_argument("--smoke-size", type=int, default=20)
    parser.add_argument("--main-size", type=int, default=None)
    parser.add_argument("--auto-install", action="store_true")
    parser.add_argument("--continue-on-judge-error", action="store_true")
    parser.add_argument("--run-all-methods", action="store_true")
    args = parser.parse_args()

    config = load_yaml(ROOT / args.config)
    main_size = args.main_size or int(config.get("experiment", {}).get("sample_size", 500) or 500)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    if args.continue_on_judge_error:
        env["HOTPOTQA_CONTINUE_ON_JUDGE_ERROR"] = "1"

    smoke_run = ROOT / run_dir(config, "smoke")
    main_run = ROOT / run_dir(config, "main")
    smoke_trace = smoke_run / "pipeline_trace.jsonl"
    main_trace = main_run / "pipeline_trace.jsonl"
    smoke_retry_path = smoke_run / "smoke_retries.jsonl"

    final_failures: list[str] = []
    final_warnings: dict[str, Any] = {}
    smoke_passed = False
    for attempt in range(3):
        try:
            prepare_build_run_eval(args.config, "smoke", args.smoke_size, "reports/HOTPOTQA_ROUND1_SMOKE_REPORT.md", smoke_trace, env)
            smoke_passed, final_failures, final_warnings = smoke_gate(config, smoke_run)
            if smoke_passed:
                break
            append_jsonl(
                smoke_retry_path,
                {"attempt": attempt + 1, "passed": False, "failures": final_failures, "warnings": final_warnings},
            )
        except Exception as exc:
            final_failures = [f"{type(exc).__name__}: {exc}"]
            final_warnings = {"attempt": attempt + 1}
            append_jsonl(smoke_retry_path, {"attempt": attempt + 1, "passed": False, "failures": final_failures})
        if attempt < 2:
            print(f"Smoke gate failed on attempt {attempt + 1}; retrying.")

    if not smoke_passed:
        write_failure_report(ROOT / "reports/HOTPOTQA_ROUND1_SMOKE_FAILURE.md", final_failures, final_warnings)
        return 1

    update_manifest(smoke_run, {"smoke_passed": True, "full_run_triggered": True})
    prepare_build_run_eval(args.config, "main", main_size, "reports/HOTPOTQA_ROUND1_FULL_REPORT.md", main_trace, env)
    update_manifest(main_run, {"smoke_passed": True, "full_run_completed": True, "main_size": main_size})
    run_stage([sys.executable, "scripts/evaluate_hotpotqa_results.py", "--run-dir", "runs/hotpotqa_round1_main", "--write-report", "reports/HOTPOTQA_ROUND1_FULL_REPORT.md"], main_trace, env=env)
    run_stage([sys.executable, "-m", "compileall", "."], main_trace, env=env)

    report = ROOT / "reports/HOTPOTQA_ROUND1_FULL_REPORT.md"
    if report.exists():
        shutil.copy2(report, ROOT / "reports/HOTPOTQA_ROUND1_REPORT.md")
    print("HotpotQA Round 1 pipeline completed: smoke passed and full run finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
