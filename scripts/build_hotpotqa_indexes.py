#!/usr/bin/env python
"""Build TextRAG, VectorRAG, and GraphRAG indexes for HotpotQA."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.graph_index import build_graph_index
from src.indexing.text_index import TextIndex
from src.indexing.vector_index import VectorIndex
from src.io_utils import index_dir, load_yaml, processed_dir, read_jsonl, write_json


def main() -> int:
    """Build and persist all HotpotQA indexes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa_round1.yaml")
    parser.add_argument("--sample-suffix", default=None)
    args = parser.parse_args()

    config = load_yaml(ROOT / args.config)
    data_dir = ROOT / processed_dir(config, args.sample_suffix)
    chunks = read_jsonl(data_dir / "chunks.jsonl")
    if not chunks:
        raise RuntimeError(f"No chunks found at {data_dir / 'chunks.jsonl'}; run prepare_hotpotqa.py first.")

    started = time.perf_counter()
    text_out = ROOT / index_dir(config, "text_index_dir", args.sample_suffix)
    vector_out = ROOT / index_dir(config, "vector_index_dir", args.sample_suffix)
    graph_out = ROOT / index_dir(config, "graph_index_dir", args.sample_suffix)

    text_index = TextIndex.build(chunks)
    text_index.save(text_out)
    vector_index = VectorIndex.build(chunks, backend=str(config.get("indexing", {}).get("vector_backend", "tfidf_char")))
    vector_index.save(vector_out)
    graph_metrics = build_graph_index(chunks, graph_out)

    manifest = {
        "sample_suffix": args.sample_suffix,
        "processed_dir": str(data_dir),
        "num_chunks": len(chunks),
        "text_index_dir": str(text_out),
        "text_backend": text_index.backend,
        "vector_index_dir": str(vector_out),
        "vector_backend": vector_index.backend,
        "graph_index_dir": str(graph_out),
        "graph_metrics": graph_metrics,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(ROOT / index_dir(config, "graph_index_dir", args.sample_suffix) / "index_manifest.json", manifest)

    if args.sample_suffix == "main":
        for key, source in [("text_index_dir", text_out), ("vector_index_dir", vector_out), ("graph_index_dir", graph_out)]:
            base = ROOT / index_dir(config, key, None)
            if base.resolve() != source.resolve():
                if base.exists():
                    shutil.rmtree(base)
                shutil.copytree(source, base)
    print(f"Built HotpotQA indexes for {len(chunks)} chunks")
    print(f"Text backend: {text_index.backend}; Vector backend: {vector_index.backend}; Graph edges: {graph_metrics.get('relation_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
