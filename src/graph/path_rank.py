"""Path ranking for GraphRAG evidence selection."""

from __future__ import annotations

import re
from typing import Any


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", str(text or "")) if len(t) > 2}


def rank_paths(question: str, paths: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Rank graph paths by token overlap between question and path labels."""
    q_tokens = _tokens(question)
    scored: list[dict[str, Any]] = []
    for path in paths:
        text = " ".join(path.get("nodes", [])) + " " + " ".join(edge.get("relation", "") for edge in path.get("edges", []))
        p_tokens = _tokens(text)
        score = len(q_tokens & p_tokens) / max(1, len(q_tokens))
        score += 0.05 / max(1, len(path.get("edges", [])))
        item = dict(path)
        item["path_score"] = float(score)
        scored.append(item)
    return sorted(scored, key=lambda x: x.get("path_score", 0.0), reverse=True)[:top_k]
