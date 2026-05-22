from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from vg_graphrag.models import Edge, Node


def _tokens(text: str) -> set[str]:
    import re

    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 1}


class MemoryGraphStore:
    def __init__(self, nodes: List[Node] | None = None, edges: List[Edge] | None = None, graph_run_id: str = "memory"):
        self.graph_run_id = graph_run_id
        self.relation_type = "RELATION"
        self.allow_fallback_relation_type = False
        self.nodes: Dict[str, Node] = {n.node_id: n for n in (nodes or [])}
        self.edges: Dict[str, Edge] = {e.edge_id: e for e in (edges or [])}
        self.adj: Dict[str, List[Edge]] = defaultdict(list)
        for e in self.edges.values():
            self.adj[e.source].append(e)
            self.adj[e.target].append(Edge(e.edge_id, e.target, e.source, e.relation, e.supporting_quote, e.source_chunk_id, e.source_document_id, e.confidence, e.provenance))

    def set_relation_mode(self, relation_type: str, allow_fallback_relation_type: bool = False) -> None:
        self.relation_type = relation_type or "RELATION"
        self.allow_fallback_relation_type = bool(allow_fallback_relation_type)

    def search_entities(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]:
        qt = _tokens(query)
        ctx = _tokens(" ".join(context_terms or []))
        scored = []
        for n in self.nodes.values():
            names = [n.name] + list(n.aliases)
            nt = _tokens(" ".join(names))
            ql = (query or "").lower()
            exact = any((name or "").lower() in ql or ql in (name or "").lower() for name in names if name)
            overlap = len((qt | ctx) & nt)
            score = (3.0 if exact else 0.0) + overlap
            if n.node_type == "claim":
                score -= 0.25
            if any(tok in ql for tok in ["progesterone", "embryo", "mastitis", "ventilation", "hygiene", "social", "energy"]):
                score += 0.5 * len(qt & nt)
            if score > 0:
                scored.append({"node": n, "node_id": n.node_id, "name": n.name, "match_score": score, "provenance": n.provenance})
        scored.sort(key=lambda x: (-x["match_score"], x["name"]))
        return scored[:limit]

    def search_claims(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]:
        qt = _tokens(query)
        ctx = _tokens(" ".join(context_terms or []))
        scored = []
        for n in self.nodes.values():
            if n.node_type != "claim":
                continue
            prov = n.provenance or {}
            claim_text = str(prov.get("claim_text") or n.name or "")
            head = str(prov.get("head") or "")
            relation = str(prov.get("relation") or "")
            tail = str(prov.get("tail") or "")
            support = str(prov.get("supporting_quote") or "")
            ct = _tokens(" ".join([claim_text, head, relation, tail, support]))
            q_overlap = len(qt & ct)
            c_overlap = len(ctx & ct)
            overlap = q_overlap + (2 * c_overlap)
            if overlap <= 0:
                continue
            score = float(overlap)
            if ctx:
                ht = _tokens(" ".join([head, tail]))
                score += 1.5 * len(ctx & ht)
                score += 1.0 * len(ctx & _tokens(relation))
            if support:
                score += 0.5
            if prov.get("source_type") == "direct_qa_train":
                score += 0.5
            scored.append(
                {
                    "claim_id": n.node_id,
                    "claim_text": claim_text,
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "supporting_quote": support,
                    "source_chunk_id": prov.get("source_chunk_id"),
                    "source_document_id": prov.get("source_document_id"),
                    "source_type": prov.get("source_type"),
                    "source_question_id": prov.get("source_question_id"),
                    "confidence": prov.get("confidence"),
                    "match_score": score,
                    "provenance": prov,
                }
            )
        scored.sort(key=lambda x: (-float(x["match_score"]), str(x["claim_text"])))
        return scored[:limit]

    def neighbors(self, node_id: str, max_hops: int = 1, relation_filters: list[str] | None = None) -> dict:
        filters = {r.lower() for r in (relation_filters or [])}
        seen_nodes = {node_id}
        out_edges: dict[str, Edge] = {}
        q = deque([(node_id, 0)])
        while q:
            cur, dist = q.popleft()
            if dist >= max_hops:
                continue
            for e in self.adj.get(cur, []):
                if filters and e.relation.lower() not in filters:
                    continue
                out_edges[e.edge_id] = self.edges.get(e.edge_id, e)
                if e.target not in seen_nodes:
                    seen_nodes.add(e.target)
                    q.append((e.target, dist + 1))
        return {"nodes": [self.nodes[n] for n in seen_nodes if n in self.nodes], "edges": list(out_edges.values())}

    def paths(self, source_id: str, target_id: str, max_hops: int = 3, relation_filters: list[str] | None = None) -> List[dict]:
        filters = {r.lower() for r in (relation_filters or [])}
        paths: List[dict] = []
        q = deque([(source_id, [source_id], [])])
        while q:
            cur, nodes, edges = q.popleft()
            if len(edges) >= max_hops:
                continue
            for e in self.adj.get(cur, []):
                if filters and e.relation.lower() not in filters:
                    continue
                if e.target in nodes:
                    continue
                next_nodes = nodes + [e.target]
                real_edge = self.edges.get(e.edge_id, e)
                next_edges = edges + [real_edge]
                if e.target == target_id:
                    paths.append({"nodes": next_nodes, "edges": next_edges})
                else:
                    q.append((e.target, next_nodes, next_edges))
        return paths

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Edge | None:
        return self.edges.get(edge_id)
