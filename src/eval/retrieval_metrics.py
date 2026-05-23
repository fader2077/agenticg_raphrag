"""Retrieval ranking metrics."""

from __future__ import annotations

import math
from typing import Any


def chunk_recall_at_k(gold_titles: list[str], retrieved_chunks: list[dict[str, Any]], k: int) -> float:
    """Return title-level chunk recall@k."""
    gold = {str(t).strip() for t in gold_titles if str(t).strip()}
    if not gold:
        return 0.0
    pred = {str(c.get("title", "")).strip() for c in retrieved_chunks[:k]}
    return len(gold & pred) / len(gold)


def mrr(gold_titles: list[str], retrieved_chunks: list[dict[str, Any]]) -> float:
    """Return reciprocal rank for the first gold-title hit."""
    gold = {str(t).strip() for t in gold_titles if str(t).strip()}
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        if str(chunk.get("title", "")).strip() in gold:
            return 1.0 / idx
    return 0.0


def ndcg(gold_titles: list[str], retrieved_chunks: list[dict[str, Any]], k: int = 10) -> float:
    """Return binary nDCG@k over gold titles."""
    gold = {str(t).strip() for t in gold_titles if str(t).strip()}
    if not gold:
        return 0.0
    dcg = 0.0
    for idx, chunk in enumerate(retrieved_chunks[:k], start=1):
        rel = 1.0 if str(chunk.get("title", "")).strip() in gold else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
