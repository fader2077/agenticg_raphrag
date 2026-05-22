from __future__ import annotations

import re

from vg_graphrag.domain import build_domain_hints
from vg_graphrag.models import ChannelEvidence, SubQuery


def _best_intermediate_answer(evidence: ChannelEvidence) -> str:
    for item in evidence.relational_evidence:
        nodes = item.get("nodes") or []
        if len(nodes) >= 2:
            return str(nodes[-1])
        node = item.get("node_id") or item.get("name")
        if node:
            return str(node)
    for item in evidence.semantic_evidence:
        text = item.get("text") or (item.get("chunk") or {}).get("text", "")
        words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)
        if words:
            return " ".join(words[:4])
    return ""


def ground_subquery(subquery: SubQuery, prior_evidence: list[ChannelEvidence]) -> SubQuery:
    """QG: replace #1 / Entity#1 style references using previous evidence."""
    grounded = subquery.grounded_text or subquery.text
    for idx, ev in enumerate(prior_evidence, 1):
        val = _best_intermediate_answer(ev)
        if not val:
            continue
        grounded = grounded.replace(f"#{idx}", val).replace(f"Entity#{idx}", val)
    hints = build_domain_hints(grounded)
    hint_terms = [str(x) for x in hints.get("alias_terms", [])[:5]]
    if hint_terms:
        lowered = grounded.lower()
        if not any(term.replace("_", " ") in lowered or term in lowered for term in hint_terms[:3]):
            grounded = f"{grounded} Focus on: {', '.join(hint_terms)}."
    subquery.grounded_text = grounded
    return subquery
