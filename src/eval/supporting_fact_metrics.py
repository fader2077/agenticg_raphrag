"""Supporting fact and gold title metrics."""

from __future__ import annotations

from typing import Any


def _pairs(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(x.get("title", "")).strip(), int(x.get("sent_id", -1))) for x in items if x.get("title") is not None}


def _retrieved_pairs(chunks: list[dict[str, Any]], limit: int | None = None) -> set[tuple[str, int]]:
    selected = chunks if limit is None else chunks[:limit]
    out: set[tuple[str, int]] = set()
    for chunk in selected:
        title = str(chunk.get("title", "")).strip()
        for sid in chunk.get("sentence_ids", []):
            out.add((title, int(sid)))
    return out


def prf(pred: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    """Return precision, recall, F1 for sets."""
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    correct = len(pred & gold)
    precision = correct / len(pred) if pred else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def supporting_fact_metrics(gold_facts: list[dict[str, Any]], retrieved_chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Compute supporting fact precision/recall/F1 and recall@k."""
    gold = _pairs(gold_facts)
    pred = _retrieved_pairs(retrieved_chunks)
    precision, recall, f1 = prf(pred, gold)
    return {
        "supporting_fact_precision": precision,
        "supporting_fact_recall": recall,
        "supporting_fact_f1": f1,
        "supporting_fact_recall@5": len(_retrieved_pairs(retrieved_chunks, 5) & gold) / len(gold) if gold else 0.0,
        "supporting_fact_recall@10": len(_retrieved_pairs(retrieved_chunks, 10) & gold) / len(gold) if gold else 0.0,
        "supporting_fact_recall@20": len(_retrieved_pairs(retrieved_chunks, 20) & gold) / len(gold) if gold else 0.0,
    }


def gold_title_metrics(gold_titles: list[str], retrieved_chunks: list[dict[str, Any]]) -> dict[str, float]:
    """Compute gold title precision/recall/F1."""
    gold = {str(t).strip() for t in gold_titles if str(t).strip()}
    pred = {str(c.get("title", "")).strip() for c in retrieved_chunks if c.get("title")}
    precision, recall, f1 = prf(pred, gold)
    return {"gold_title_precision": precision, "gold_title_recall": recall, "gold_title_f1": f1}
