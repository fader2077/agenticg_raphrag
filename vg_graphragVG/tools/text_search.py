from __future__ import annotations

import re

from vg_graphrag.models import AgentState, ToolResult, to_dict
from vg_graphrag.runtime_skill import pattern_term_score
from vg_graphrag.stores.text_store import TextStore


class TextSearch:
    name = "TextSearch"

    def __init__(self, text: TextStore):
        self.text = text

    def _tokens(self, text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 2}

    def _rerank_score(self, chunk, state: AgentState) -> float:
        analysis = getattr(state, "analysis", None)
        slot = getattr(analysis, "answer_slot", "other") if analysis else "other"
        ql = (state.question or "").lower()
        text = (chunk.text or "").lower()
        score = 0.0
        canon_terms = [str(x).replace("_", " ").lower() for x in ((analysis.constraints or {}).get("canonical_terms", []) if analysis else [])]
        matched_patterns = set((((analysis.constraints or {}).get("domain_hints", {}) or {}).get("matched_patterns", []) if analysis else []))
        skill = dict((((analysis.constraints or {}) if analysis else {}).get("runtime_skill") or {}))
        for term in canon_terms:
            if term and term in text:
                score += 4.0
        score += pattern_term_score(skill, matched_patterns, text, default_boost=4.0, default_penalty=3.0)
        if slot in {"reproduction", "mechanism", "cause", "connection"}:
            if any(x in text for x in ["progesterone", "corpus luteum", "embryo", "embryonic", "uterine", "ovulatory", "ovulation"]):
                score += 6.0
            if any(x in text for x in ["profitability", "market price", "cash flow", "operating expenses", "marketable kids"]):
                score -= 7.0
        if slot in {"management", "economic"} or any(x in ql for x in ["economic returns", "cash flow", "market", "profit", "volatile"]):
            if any(x in text for x in ["planned production", "market", "auction", "cash flow", "profit", "revenue", "supply and demand", "predict operating profits"]):
                score += 6.0
            if any(x in text for x in ["reproductive efficiency", "conception rates", "estrus expression"]) and not any(x in text for x in ["market", "cash flow", "revenue", "profit"]):
                score -= 4.0
        if any(x in ql for x in ["feed intake", "diet formulation", "feeding environment"]):
            if any(x in text for x in ["feed intake", "feed bunk", "competition", "social stress", "feeding environment", "housing"]):
                score += 5.0
        return score

    def run(self, input: dict, state: AgentState) -> ToolResult:
        query = input.get("query") or state.question
        chunks = self.text.search(query, limit=int(input.get("limit", 5)))
        chunks = sorted(chunks, key=lambda c: self._rerank_score(c, state), reverse=True)
        return ToolResult(self.name, input, results=[to_dict(c) for c in chunks])
