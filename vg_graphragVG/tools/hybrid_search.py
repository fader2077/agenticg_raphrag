from __future__ import annotations

import re

from vg_graphrag.models import AgentState, ToolResult, to_dict
from vg_graphrag.runtime_skill import pattern_term_score
from vg_graphrag.stores.graph_store import GraphStore
from vg_graphrag.stores.text_store import TextStore


class HybridSearch:
    name = "HybridSearch"

    def __init__(self, graph: GraphStore, text: TextStore):
        self.graph = graph
        self.text = text

    def _rerank_score(self, chunk: dict, state: AgentState) -> float:
        analysis = getattr(state, "analysis", None)
        slot = getattr(analysis, "answer_slot", "other") if analysis else "other"
        ql = (state.question or "").lower()
        text = str((chunk.get("chunk") or {}).get("text") or "").lower()
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
        if chunk.get("linked_edge"):
            score += 1.0
        return score

    def run(self, input: dict, state: AgentState) -> ToolResult:
        query = input.get("query") or state.question
        node_ids = input.get("node_ids") or []
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        results = []
        seen = set()
        for node_id in node_ids:
            nb = self.graph.neighbors(node_id, max_hops=int(input.get("max_hops", 1)), relation_filters=input.get("relation_filters") or [])
            for e in nb["edges"]:
                if e.source_chunk_id and e.source_chunk_id not in seen:
                    c = self.text.get(e.source_chunk_id)
                    if c:
                        seen.add(c.chunk_id)
                        results.append({"chunk": to_dict(c), "linked_edge": to_dict(e)})
        for c in self.text.search(query, limit=int(input.get("limit", 5))):
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                results.append({"chunk": to_dict(c), "linked_edge": None})
        results.sort(key=lambda x: self._rerank_score(x, state), reverse=True)
        return ToolResult(self.name, input, results=results, provenance_metadata={"graph_run_id": self.graph.graph_run_id})
