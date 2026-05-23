#!/usr/bin/env python
"""Prepare normalized HotpotQA artifacts for a sample suffix."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hotpotqa_loader import choose_sample, load_hotpotqa_rows, write_processed_hotpotqa
from src.io_utils import load_yaml, processed_dir, set_seed


def main() -> int:
    """Load, sample, normalize, and write HotpotQA processed artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa_round1.yaml")
    parser.add_argument("--setting", default="distractor")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-size", default=None)
    parser.add_argument("--output-suffix", default=None)
    args = parser.parse_args()

    config = load_yaml(ROOT / args.config)
    seed = int(config.get("experiment", {}).get("random_seed", 42))
    set_seed(seed)
    sample_size = args.sample_size if args.sample_size is not None else config.get("experiment", {}).get("sample_size", 500)
    rows, source_meta = load_hotpotqa_rows(args.setting, args.split)
    sampled = choose_sample(rows, sample_size, seed)
    out_dir = ROOT / processed_dir(config, args.output_suffix)
    manifest = write_processed_hotpotqa(
        out_dir,
        sampled,
        source_meta,
        args.setting,
        sample_size,
        seed,
        split=args.split,
        max_sentences_per_chunk=int(config.get("indexing", {}).get("max_sentences_per_chunk", 3)),
        overlap_sentences=int(config.get("indexing", {}).get("overlap_sentences", 1)),
    )
    if args.output_suffix == "main":
        base_dir = ROOT / processed_dir(config, None)
        if base_dir.resolve() != out_dir.resolve():
            base_dir.mkdir(parents=True, exist_ok=True)
            for file in out_dir.glob("*"):
                if file.is_file():
                    shutil.copy2(file, base_dir / file.name)
    print(f"Prepared HotpotQA artifacts at {out_dir} ({manifest['num_questions']} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
