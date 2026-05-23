#!/usr/bin/env python
"""Evaluate HotpotQA Round 1 outputs and optionally write a report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import evaluate_run_dir, write_report
from src.io_utils import read_json


def main() -> int:
    """Evaluate a run directory and write aggregate artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-report", default=None)
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    all_metrics = evaluate_run_dir(run_dir)
    if args.write_report:
        manifest = read_json(run_dir / "experiment_manifest.json", {}) or {}
        write_report(run_dir, ROOT / args.write_report, all_metrics, smoke_passed=manifest.get("smoke_passed"))
    print(f"Evaluated {len(all_metrics)} methods under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
