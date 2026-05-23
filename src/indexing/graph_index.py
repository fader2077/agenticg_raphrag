"""Build the HotpotQA local graph index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.graph.graph_store import LocalGraphStore
from src.graph.relation_extraction import extract_relations_from_chunk
from src.io_utils import write_json


def build_graph_index(chunks: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Build a provenance-linked NetworkX/JSON graph index from chunks."""
    store = LocalGraphStore()
    rejected = 0
    for chunk in chunks:
        store.add_chunk(chunk)
        for edge in extract_relations_from_chunk(chunk):
            if not store.add_edge(edge):
                rejected += 1
    store.save(output_dir)
    metrics = store.audit()
    metrics["rejected_edges"] = rejected
    metrics["neo4j_status"] = "not_used_fallback_networkx_json"
    write_json(output_dir / "graph_metrics.json", metrics)
    return metrics
