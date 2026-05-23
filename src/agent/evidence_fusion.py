"""Evidence fusion for bounded AgenticGraphRAG."""

from __future__ import annotations

from typing import Any


def fuse_evidence(*chunk_lists: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Deduplicate chunks across routes while preserving best scores and order."""
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for chunks in chunk_lists:
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            if cid not in merged:
                merged[cid] = dict(chunk)
                order.append(cid)
            else:
                merged[cid]["score"] = max(float(merged[cid].get("score", 0.0)), float(chunk.get("score", 0.0)))
                merged[cid]["source"] = f"{merged[cid].get('source')}+{chunk.get('source')}"
    return [merged[cid] for cid in order[:limit]]
