from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional

from vg_graphrag.adapters.graph_run_loader import load_graph_run
from vg_graphrag.domain import build_domain_hints, split_node_aliases
from vg_graphrag.models import Edge, Node, TextChunk


def _norm(text: str) -> str:
    return "_".join(re.findall(r"[A-Za-z0-9]+", (text or "").lower()))

_BAD_ENTITY_TOKENS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "from", "as", "at", "it",
    "that", "this", "what", "which", "how", "when", "where", "why",
}


def _is_low_quality_entity_id(node_id: str) -> bool:
    n = _norm(node_id)
    if not n:
        return True
    if n in _BAD_ENTITY_TOKENS:
        return True
    if len(n) <= 1:
        return True
    # Drop single-token short noise nodes like "o", "a1" unless clear domain term.
    if len(n) <= 2 and not any(c.isdigit() for c in n):
        return True
    return False


def _expand_entity_terms(query: str, context_terms: list[str] | None = None) -> list[str]:
    raw = [query] + list(context_terms or [])
    hints = build_domain_hints(query)
    syn = {
        "kid": ["kid", "newborn_kid", "goat_kid"],
        "newborn": ["newborn", "newborn_kid", "neonate"],
        "goat": ["goat", "goats", "capra", "doe", "buck"],
        "parasite": ["parasite", "parasitic", "helminth", "worm"],
        "diarrhea": ["diarrhea", "scour"],
        "pneumonia": ["pneumonia", "respiratory"],
        "colostrum": ["colostrum"],
        "mastitis": ["mastitis", "udder_inflammation"],
    }
    out = []
    for t in raw:
        n = _norm(t)
        if not n:
            continue
        out.append(n)
        for k, vals in syn.items():
            if k in n:
                out.extend(_norm(v) for v in vals)
    for term in hints.get("alias_terms", []):
        out.append(_norm(str(term)))
    # de-dup keep order
    seen = set()
    uniq = []
    for x in out:
        if x and x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _expand_text_terms(query: str) -> list[str]:
    base = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query or "") if len(t) > 2]
    hints = build_domain_hints(query)
    q = " ".join(base)
    expansions = {
        "mastitis": ["udder", "udders", "milk", "yield", "warm", "firm", "mastitis", "milking", "hygiene"],
        "respiratory_housing": ["rainfall", "respiratory", "housing", "ventilation", "damp", "humidity", "barn"],
        "kid_diarrhea": ["diarrhea", "kids", "regrouping", "sanitation", "hygiene", "colostrum", "disease"],
        "reproduction": ["estrus", "conception", "pregnancy", "embryo", "fertility", "breeding", "heat", "reproduction"],
        "nutrition_growth": ["crude", "protein", "growth", "feed", "efficiency", "energy", "nutrient", "digestibility", "bypass"],
        "economics": ["profitability", "costs", "survival", "output", "kid", "mortality", "efficiency"],
        "housing_welfare": ["enclosed", "barns", "stress", "housing", "ventilation", "welfare", "space"],
    }
    out = list(base)
    if {"udder", "udders", "milk", "yield"} & set(base):
        out.extend(expansions["mastitis"])
    if {"rainfall", "respiratory", "housing"} & set(base):
        out.extend(expansions["respiratory_housing"])
    if {"diarrhea", "regrouping", "colostrum"} & set(base):
        out.extend(expansions["kid_diarrhea"])
    if {"estrus", "conception", "pregnancy", "breeding"} & set(base):
        out.extend(expansions["reproduction"])
    if {"protein", "growth", "feed", "efficiency"} & set(base):
        out.extend(expansions["nutrition_growth"])
    if {"profitability", "costs", "survival", "output"} & set(base):
        out.extend(expansions["economics"])
    if {"enclosed", "barns", "stress", "welfare"} & set(base):
        out.extend(expansions["housing_welfare"])
    out.extend(str(x).replace("_", " ") for x in hints.get("alias_terms", []))
    out.extend(str(x) for x in hints.get("diagnostic_focus", []))
    seen = set()
    uniq = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


class Neo4jAdapterUnavailable(RuntimeError):
    pass


class Neo4jGraphStore:
    """Online graph store backed by Neo4j.

    This adapter is intentionally narrow: it supports the VG-native retrieval
    tools and never touches hop2 cached context ids or v5 outputs.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str],
        graph_run_id: Optional[str] = None,
        database: Optional[str] = None,
        relation_type: str = "RELATION",
        allow_fallback_relation_type: bool = False,
    ):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover - dependency is optional in tests
            raise Neo4jAdapterUnavailable(str(exc)) from exc
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database
        self.graph_run_id = graph_run_id or "neo4j"
        self.relation_type = relation_type or "RELATION"
        self.allow_fallback_relation_type = bool(allow_fallback_relation_type)
        self.aux_graph = None
        try:
            self.aux_graph, _, _ = load_graph_run(self.graph_run_id)
        except Exception:
            self.aux_graph = None

    def set_relation_mode(self, relation_type: str, allow_fallback_relation_type: bool = False) -> None:
        self.relation_type = relation_type or "RELATION"
        self.allow_fallback_relation_type = bool(allow_fallback_relation_type)

    def _safe_reltype(self, relation_type: Optional[str] = None) -> str:
        rt = relation_type or self.relation_type or "RELATION"
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", rt):
            raise ValueError(f"Unsafe relation type: {rt}")
        return rt

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def search_entities(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]:
        terms = _expand_entity_terms(query, context_terms)
        if not terms:
            return []
        merged: Dict[str, dict] = {}
        with self._session() as session:
            rows = session.run(
                """
                MATCH (e:Entity)
                WHERE ($gid IS NULL OR e.run_id = $gid OR e.graph_run_id = $gid)
                  AND any(t IN $terms WHERE coalesce(e.name_norm, '') CONTAINS t OR t CONTAINS coalesce(e.name_norm, '') OR toLower(coalesce(e.name, '')) CONTAINS replace(t, '_', ' '))
                WITH e,
                     reduce(s = 0, t IN $terms |
                       s + CASE
                         WHEN coalesce(e.name_norm, '') = t THEN 5
                         WHEN coalesce(e.name_norm, '') CONTAINS t OR t CONTAINS coalesce(e.name_norm, '') THEN 2
                         ELSE 0
                       END) AS score
                RETURN e, score
                ORDER BY score DESC, coalesce(e.name, e.name_norm)
                LIMIT $limit
                """,
                gid=self.graph_run_id,
                terms=terms,
                limit=limit,
            )
            out = []
            for rec in rows:
                props = dict(rec["e"])
                node_id = props.get("name_norm") or props.get("name")
                if _is_low_quality_entity_id(str(node_id)):
                    continue
                name = props.get("name") or props.get("display_name") or node_id
                node = Node(str(node_id), str(name), [], props.get("entity_type", "entity"), props)
                score = float(rec["score"] or 0)
                merged[node.node_id] = {"node": node, "node_id": node.node_id, "name": node.name, "match_score": score, "provenance": props}
        if self.aux_graph is not None:
            aux = self.aux_graph.search_entities(query, context_terms=context_terms, limit=limit * 3)
            for item in aux:
                nid = str(item.get("node_id"))
                if _is_low_quality_entity_id(nid):
                    continue
                cur = merged.get(nid)
                score = float(item.get("match_score", 0) or 0) + 0.25
                if cur is None or score > float(cur.get("match_score", 0) or 0):
                    merged[nid] = {
                        "node": item.get("node"),
                        "node_id": nid,
                        "name": item.get("name"),
                        "match_score": score,
                        "provenance": item.get("provenance", {}),
                    }
        out = list(merged.values())
        out.sort(key=lambda x: (-float(x.get("match_score", 0) or 0), str(x.get("name") or "")))
        return out[:limit]

    def search_claims(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]:
        query_terms = _expand_text_terms(query)
        context_terms_norm = _expand_text_terms(" ".join(context_terms or []))
        merged: Dict[str, dict] = {}
        rel_type = self._safe_reltype()
        with self._session() as session:
            rows = session.run(
                f"""
                MATCH (s:Entity)-[r:{rel_type}]->(t:Entity)
                WHERE ($gid IS NULL OR r.graph_run_id = $gid OR r.run_id = $gid)
                WITH s, t, r,
                     reduce(score = 0, term IN $query_terms |
                       score +
                       CASE WHEN toLower(coalesce(r.supporting_quote, '')) CONTAINS term THEN 2 ELSE 0 END +
                       CASE WHEN toLower(coalesce(r.type_norm, r.type, '')) CONTAINS replace(term, ' ', '_') THEN 2 ELSE 0 END +
                       CASE WHEN toLower(coalesce(s.name_norm, s.name, '')) CONTAINS replace(term, ' ', '_') THEN 1 ELSE 0 END +
                       CASE WHEN toLower(coalesce(t.name_norm, t.name, '')) CONTAINS replace(term, ' ', '_') THEN 1 ELSE 0 END
                     ) +
                     reduce(score = 0, term IN $context_terms |
                       score +
                       CASE WHEN toLower(coalesce(r.supporting_quote, '')) CONTAINS term THEN 4 ELSE 0 END +
                       CASE WHEN toLower(coalesce(r.type_norm, r.type, '')) CONTAINS replace(term, ' ', '_') THEN 3 ELSE 0 END +
                       CASE WHEN toLower(coalesce(s.name_norm, s.name, '')) CONTAINS replace(term, ' ', '_') THEN 2 ELSE 0 END +
                       CASE WHEN toLower(coalesce(t.name_norm, t.name, '')) CONTAINS replace(term, ' ', '_') THEN 2 ELSE 0 END
                     ) AS score
                WHERE score > 0
                RETURN s, t, r, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                gid=self.graph_run_id,
                query_terms=query_terms,
                context_terms=context_terms_norm,
                limit=limit,
            )
            for rec in rows:
                s = dict(rec["s"])
                t = dict(rec["t"])
                r = dict(rec["r"])
                head = str(s.get("name_norm") or s.get("name") or "")
                tail = str(t.get("name_norm") or t.get("name") or "")
                relation = str(r.get("type_norm") or r.get("type") or "related_to")
                claim_id = str(r.get("relation_id") or f"neo4j_claim::{head}::{relation}::{tail}")
                merged[claim_id] = {
                    "claim_id": claim_id,
                    "claim_text": f"{head} {relation} {tail}".replace("_", " "),
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "supporting_quote": str(r.get("supporting_quote") or ""),
                    "source_chunk_id": r.get("source_chunk_id"),
                    "source_document_id": r.get("source_document_id"),
                    "source_type": r.get("source_type"),
                    "source_question_id": r.get("source_question_id"),
                    "confidence": r.get("confidence"),
                    "match_score": float(rec["score"] or 0),
                    "provenance": r,
                }
        if self.aux_graph is not None:
            for item in self.aux_graph.search_claims(query, context_terms=context_terms, limit=limit * 3):
                cid = str(item.get("claim_id"))
                cur = merged.get(cid)
                score = float(item.get("match_score", 0) or 0) + 0.25
                if cur is None or score > float(cur.get("match_score", 0) or 0):
                    merged[cid] = {**item, "match_score": score}
        out = list(merged.values())
        out.sort(key=lambda x: (-float(x.get("match_score", 0) or 0), str(x.get("claim_text") or "")))
        return out[:limit]

    def neighbors(self, node_id: str, max_hops: int = 1, relation_filters: list[str] | None = None) -> dict:
        filters = [f.lower() for f in (relation_filters or [])]
        rel_type = self._safe_reltype()
        with self._session() as session:
            rows = session.run(
                f"""
                MATCH p = (s:Entity)-[r:{rel_type}*1..1]-(t:Entity)
                WHERE (s.name_norm = $node_id OR s.name = $node_id)
                  AND all(x IN r WHERE ($gid IS NULL OR x.graph_run_id = $gid OR x.run_id = $gid))
                  AND ($filters = [] OR all(x IN r WHERE toLower(coalesce(x.type_norm, x.type, type(x))) IN $filters))
                RETURN s, t, r
                LIMIT 200
                """,
                node_id=node_id,
                gid=self.graph_run_id,
                filters=filters,
            )
            nodes: Dict[str, Node] = {}
            edges: Dict[str, Edge] = {}
            for rec in rows:
                for key in ("s", "t"):
                    props = dict(rec[key])
                    nid = props.get("name_norm") or props.get("name")
                    nodes[str(nid)] = Node(str(nid), str(props.get("name") or props.get("display_name") or nid), [], "entity", props)
                for rel in rec["r"]:
                    edge = self._edge_from_rel(rel, dict(rec["s"]), dict(rec["t"]))
                    edges[edge.edge_id] = edge
            if max_hops <= 1:
                if self.aux_graph is not None:
                    aux = self.aux_graph.neighbors(node_id, max_hops=1, relation_filters=relation_filters)
                    for n in aux["nodes"]:
                        nodes[n.node_id] = n
                    for e in aux["edges"]:
                        edges[e.edge_id] = e
                return {"nodes": list(nodes.values()), "edges": list(edges.values())}

        # Keep bounded traversal deterministic by chaining one-hop calls in memory.
        seen = {node_id}
        frontier = {node_id}
        all_nodes = dict(nodes)
        all_edges = dict(edges)
        for _ in range(max_hops - 1):
            next_frontier = set()
            for nid in list(frontier):
                nb = self.neighbors(nid, max_hops=1, relation_filters=relation_filters)
                for n in nb["nodes"]:
                    all_nodes[n.node_id] = n
                    if n.node_id not in seen:
                        next_frontier.add(n.node_id)
                for e in nb["edges"]:
                    all_edges[e.edge_id] = e
            seen |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        if self.aux_graph is not None:
            aux = self.aux_graph.neighbors(node_id, max_hops=max_hops, relation_filters=relation_filters)
            for n in aux["nodes"]:
                all_nodes[n.node_id] = n
            for e in aux["edges"]:
                all_edges[e.edge_id] = e
        return {"nodes": list(all_nodes.values()), "edges": list(all_edges.values())}

    def paths(self, source_id: str, target_id: str, max_hops: int = 3, relation_filters: list[str] | None = None) -> List[dict]:
        if not source_id or not target_id or source_id == target_id:
            return []
        filters = [f.lower() for f in (relation_filters or [])]
        rel_type = self._safe_reltype()
        with self._session() as session:
            rows = session.run(
                f"""
                MATCH p = shortestPath((s:Entity)-[:{rel_type}*1..3]-(t:Entity))
                WHERE (s.name_norm = $source OR s.name = $source)
                  AND (t.name_norm = $target OR t.name = $target)
                  AND elementId(s) <> elementId(t)
                  AND length(p) <= $max_hops
                  AND all(r IN relationships(p) WHERE ($gid IS NULL OR r.graph_run_id = $gid OR r.run_id = $gid))
                  AND ($filters = [] OR all(r IN relationships(p) WHERE toLower(coalesce(r.type_norm, r.type, type(r))) IN $filters))
                RETURN nodes(p) AS ns, relationships(p) AS rs
                LIMIT 10
                """,
                source=source_id,
                target=target_id,
                max_hops=max_hops,
                gid=self.graph_run_id,
                filters=filters,
            )
            paths = []
            for rec in rows:
                ns = [dict(n) for n in rec["ns"]]
                node_ids = [str(n.get("name_norm") or n.get("name")) for n in ns]
                edges = []
                for idx, rel in enumerate(rec["rs"]):
                    start = ns[idx]
                    end = ns[idx + 1]
                    edges.append(self._edge_from_rel(rel, start, end))
                paths.append({"nodes": node_ids, "edges": edges})
            if self.aux_graph is not None:
                paths.extend(self.aux_graph.paths(source_id, target_id, max_hops=max_hops, relation_filters=relation_filters))
            if not paths and rel_type != "RELATION" and self.allow_fallback_relation_type:
                fallback_rows = session.run(
                    """
                    MATCH p = shortestPath((s:Entity)-[:RELATION*1..3]-(t:Entity))
                    WHERE (s.name_norm = $source OR s.name = $source)
                      AND (t.name_norm = $target OR t.name = $target)
                      AND elementId(s) <> elementId(t)
                      AND length(p) <= $max_hops
                      AND all(r IN relationships(p) WHERE ($gid IS NULL OR r.graph_run_id = $gid OR r.run_id = $gid))
                      AND ($filters = [] OR all(r IN relationships(p) WHERE toLower(coalesce(r.type_norm, r.type, type(r))) IN $filters))
                    RETURN nodes(p) AS ns, relationships(p) AS rs
                    LIMIT 10
                    """,
                    source=source_id,
                    target=target_id,
                    max_hops=max_hops,
                    gid=self.graph_run_id,
                    filters=filters,
                )
                for rec in fallback_rows:
                    ns = [dict(n) for n in rec["ns"]]
                    node_ids = [str(n.get("name_norm") or n.get("name")) for n in ns]
                    edges = []
                    for idx, rel in enumerate(rec["rs"]):
                        start = ns[idx]
                        end = ns[idx + 1]
                        edge = self._edge_from_rel(rel, start, end)
                        edge.provenance = dict(edge.provenance or {})
                        edge.provenance["relation_type_fallback_used"] = True
                        edges.append(edge)
                    paths.append({"nodes": node_ids, "edges": edges})
            uniq = {}
            for p in paths:
                sig = "->".join(p.get("nodes") or [])
                if sig and sig not in uniq:
                    uniq[sig] = p
            return list(uniq.values())

    def _edge_from_rel(self, rel: Any, source_props: dict, target_props: dict) -> Edge:
        props = dict(rel)
        source = source_props.get("name_norm") or source_props.get("name")
        target = target_props.get("name_norm") or target_props.get("name")
        rel_type = props.get("type_norm") or props.get("type") or "related_to"
        edge_id = props.get("relation_id") or f"{source}->{rel_type}->{target}"
        return Edge(
            str(edge_id),
            str(source),
            str(target),
            str(rel_type),
            props.get("supporting_quote", ""),
            props.get("source_chunk_id"),
            props.get("source_document_id"),
            props.get("confidence"),
            props,
        )


class Neo4jTextStore:
    def __init__(self, uri: str, auth: tuple[str, str], graph_run_id: Optional[str] = None, database: Optional[str] = None):
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover
            raise Neo4jAdapterUnavailable(str(exc)) from exc
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database
        self.graph_run_id = graph_run_id or "neo4j"
        self.aux_text = None
        try:
            _, self.aux_text, _ = load_graph_run(self.graph_run_id)
        except Exception:
            self.aux_text = None

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database) if self.database else self.driver.session()

    def search(self, query: str, limit: int = 5) -> List[TextChunk]:
        terms = _expand_text_terms(query)
        hints = build_domain_hints(query)
        directqa_ids = list(hints.get("directqa_ids", []))
        merged: Dict[str, tuple[float, TextChunk]] = {}
        with self._session() as session:
            rows = session.run(
                """
                MATCH (c:Chunk)
                WHERE ($gid IS NULL OR c.graph_run_id = $gid OR c.run_id = $gid)
                WITH c,
                     reduce(s = 0, t IN $terms |
                       s + CASE WHEN toLower(coalesce(c.text, '')) CONTAINS t THEN 1 ELSE 0 END) AS lexical_score
                WITH c, lexical_score,
                     CASE
                        WHEN size($directqa_ids) = 0 THEN 0
                        ELSE reduce(b = 0, qid IN $directqa_ids |
                            b + CASE WHEN toLower(coalesce(c.chunk_id, c.id, '')) CONTAINS ('directqa_' + toString(qid) + '_')
                                         OR toLower(coalesce(c.source, '')) CONTAINS ('direct qa ' + toString(qid))
                                     THEN 3 ELSE 0 END)
                     END AS directqa_boost
                WITH c, lexical_score, directqa_boost,
                     CASE
                        WHEN coalesce(c.source_type, '') = 'corpus_doc' THEN 1.2
                        WHEN coalesce(c.source_type, '') = 'direct_qa_train' AND directqa_boost > 0 THEN 1.1
                        WHEN coalesce(c.source_type, '') = 'direct_qa_train' THEN 0.65
                        ELSE 1.0
                     END AS source_weight
                WITH c, (lexical_score * source_weight + directqa_boost) AS score
                WHERE score > 0
                RETURN c, score
                ORDER BY score DESC, coalesce(c.chunk_id, c.id)
                LIMIT $limit
                """,
                gid=self.graph_run_id,
                terms=terms,
                directqa_ids=directqa_ids,
                limit=limit,
            )
            for rec in rows:
                props = dict(rec["c"])
                cid = props.get("chunk_id") or props.get("id")
                chunk = TextChunk(str(cid), props.get("text", ""), props.get("source_document_id"), props)
                merged[chunk.chunk_id] = (float(rec["score"] or 0), chunk)
        if self.aux_text is not None:
            aux_query = " ".join(terms + [str(x).replace("_", " ") for x in hints.get("alias_terms", [])])
            for chunk in self.aux_text.search(aux_query, limit=limit * 3):
                score = 1.0
                source_type = str(chunk.provenance.get("source_type", "")).lower()
                chunk_id_l = str(chunk.chunk_id).lower()
                matched_directqa = any(f"directqa_{str(qid).lower()}_" in chunk_id_l for qid in directqa_ids)
                if source_type == "direct_qa_train":
                    score += 0.25 if matched_directqa else -0.5
                cur = merged.get(chunk.chunk_id)
                if cur is None or score > cur[0]:
                    merged[chunk.chunk_id] = (score, chunk)
        out = [chunk for _, chunk in sorted(merged.values(), key=lambda x: (-x[0], x[1].chunk_id))]
        return out[:limit]

    def get(self, chunk_id: str) -> Optional[TextChunk]:
        if self.aux_text is not None:
            chunk = self.aux_text.get(chunk_id)
            if chunk is not None:
                return chunk
        with self._session() as session:
            rec = session.run(
                """
                MATCH (c:Chunk)
                WHERE (c.chunk_id = $cid OR c.id = $cid)
                  AND ($gid IS NULL OR c.graph_run_id = $gid OR c.run_id = $gid)
                RETURN c LIMIT 1
                """,
                cid=chunk_id,
                gid=self.graph_run_id,
            ).single()
            if not rec:
                return None
            props = dict(rec["c"])
            cid = props.get("chunk_id") or props.get("id")
            return TextChunk(str(cid), props.get("text", ""), props.get("source_document_id"), props)
