"""Graph retrieval metrics."""

from __future__ import annotations

from typing import Any


def graph_case_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Compute per-case graph metrics from a prediction trace."""
    paths = record.get("retrieved_paths") or []
    gold_titles = {str(t).strip() for t in record.get("gold_titles", []) or []}
    path_titles = {
        str(edge.get("source_title", "") or "").strip()
        for path in paths
        for edge in path.get("edges", [])
        if edge.get("source_title")
    }
    if not path_titles:
        path_titles = {
            str(chunk.get("title", "")).strip()
            for chunk in record.get("retrieved_chunks", [])
            if chunk.get("source") == "graph_provenance"
        }
    path_recall = len(gold_titles & path_titles) / len(gold_titles) if gold_titles else 0.0
    avg_score = sum(float(p.get("path_score", 0.0)) for p in paths) / len(paths) if paths else 0.0
    return {
        "path_found": 1.0 if paths else 0.0,
        "path_found_rate": 1.0 if paths else 0.0,
        "path_recall": path_recall,
        "avg_path_score": avg_score,
        "entity_link_success_rate": 1.0 if record.get("retrieved_entities") else 0.0,
        "retrieved_entity_count": float(len(record.get("retrieved_entities") or [])),
        "retrieved_path_count": float(len(paths)),
    }
