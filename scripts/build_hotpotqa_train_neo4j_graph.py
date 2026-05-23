#!/usr/bin/env python
"""Build or reuse the full HotpotQA-train Neo4j graph."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.indexing.neo4j_hotpotqa_graph import DEFAULT_FULLTEXT_INDEX_NAME, DEFAULT_GRAPH_RUN_ID, build_hotpotqa_train_graph


def main() -> int:
    """CLI wrapper for the Neo4j HotpotQA train graph builder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-run-id", default=DEFAULT_GRAPH_RUN_ID)
    parser.add_argument("--fulltext-index-name", default=DEFAULT_FULLTEXT_INDEX_NAME)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-docs", type=int, default=None, help="Debug only. Omit for full train graph.")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-edges-per-chunk", type=int, default=2)
    args = parser.parse_args()
    manifest = build_hotpotqa_train_graph(
        graph_run_id=args.graph_run_id,
        fulltext_index_name=args.fulltext_index_name,
        force_rebuild=args.force_rebuild,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
        max_edges_per_chunk=args.max_edges_per_chunk,
    )
    print(f"graph_run_id={manifest.get('graph_run_id')} status={manifest.get('status')} counts={manifest.get('counts')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
