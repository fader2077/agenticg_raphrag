"""Neo4j HotpotQA retrievers over a fixed train graph_run_id."""

from __future__ import annotations

import re
import time
from typing import Any

from neo4j import GraphDatabase

from src.graph.path_rank import rank_paths
from src.indexing.neo4j_hotpotqa_graph import (
    DEFAULT_ENTITY_FULLTEXT_INDEX_NAME,
    DEFAULT_FULLTEXT_INDEX_NAME,
    lucene_escape,
    neo4j_auth_from_env,
)


def _terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", str(text or "")) if len(t) > 2][:16]


class Neo4jHotpotQAStore:
    """Read-only retriever facade for one persisted HotpotQA graph_run_id."""

    def __init__(
        self,
        graph_run_id: str,
        fulltext_index_name: str = DEFAULT_FULLTEXT_INDEX_NAME,
        entity_fulltext_index_name: str = DEFAULT_ENTITY_FULLTEXT_INDEX_NAME,
    ):
        uri, auth = neo4j_auth_from_env()
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.graph_run_id = graph_run_id
        self.fulltext_index_name = fulltext_index_name
        self.entity_fulltext_index_name = entity_fulltext_index_name

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def text_search(self, question: str, top_k: int = 10, source: str = "text_search") -> list[dict[str, Any]]:
        """Search train chunks through the Neo4j fulltext index."""
        query_text = lucene_escape(question)
        rows: list[dict[str, Any]] = []
        with self.driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.fulltext.queryNodes($index_name, $query_text, {{limit: $limit * 4}}) YIELD node, score
                WHERE node.graph_run_id = $gid
                RETURN node, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                index_name=self.fulltext_index_name,
                query_text=query_text,
                gid=self.graph_run_id,
                limit=top_k,
            )
            for rank, rec in enumerate(result, start=1):
                props = dict(rec["node"])
                rows.append(_chunk_from_props(props, float(rec["score"] or 0.0), rank, source))
        return rows

    def vector_search(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        """VectorRAG fallback: fulltext retrieval when full-train embeddings are unavailable."""
        return self.text_search(question, top_k=top_k, source="vector_search_fallback_neo4j_fulltext")

    def entity_search(self, question: str, limit: int = 8) -> list[dict[str, Any]]:
        """Link question mentions to train graph entities."""
        terms = _terms(question)
        if not terms:
            return []
        with self.driver.session() as session:
            try:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes($index_name, $query_text, {limit: $candidate_limit}) YIELD node, score
                    WHERE node.graph_run_id = $gid
                    RETURN node AS e, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    index_name=self.entity_fulltext_index_name,
                    query_text=lucene_escape(question),
                    candidate_limit=max(limit * 10, 50),
                    gid=self.graph_run_id,
                    limit=limit,
                )
                rows = list(result)
            except Exception:
                result = session.run(
                    """
                    MATCH (e:HotpotEntity)
                    WHERE e.graph_run_id = $gid AND e.name_norm IN $terms
                    RETURN e, 1.0 AS score
                    LIMIT $limit
                    """,
                    gid=self.graph_run_id,
                    terms=[t.replace(" ", "_") for t in terms],
                    limit=limit,
                )
                rows = list(result)
            return [
                {
                    "node_id": dict(r["e"]).get("name_norm"),
                    "entity_key": dict(r["e"]).get("entity_key"),
                    "name": dict(r["e"]).get("name") or dict(r["e"]).get("display_name"),
                    "score": float(r["score"] or 0.0),
                }
                for r in rows
            ]

    def graph_retrieve(self, question: str, depth: int = 1, top_k_paths: int = 5, max_nodes_per_hop: int = 10) -> dict[str, Any]:
        """Retrieve graph paths and provenance chunks from Neo4j."""
        seeds = self.entity_search(question, limit=max_nodes_per_hop)
        seed_keys = [s.get("entity_key") or f"{self.graph_run_id}::{s['node_id']}" for s in seeds if s.get("node_id")]
        paths: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        entities = [s.get("name") or s.get("node_id") for s in seeds]
        if seed_keys:
            rel_depth = max(1, min(int(depth), 2))
            with self.driver.session() as session:
                for skey in seed_keys[:max_nodes_per_hop]:
                    if rel_depth == 1:
                        cypher = """
                            MATCH (s:HotpotEntity {entity_key:$skey})-[r:HOTPOT_RELATION]-(t:HotpotEntity)
                            WHERE r.graph_run_id=$gid
                            RETURN [s, t] AS ns, [r] AS rs
                            ORDER BY coalesce(r.confidence, 0.0) DESC
                            LIMIT $limit
                        """
                    else:
                        cypher = """
                            MATCH (s:HotpotEntity {entity_key:$skey})-[r1:HOTPOT_RELATION]-(m:HotpotEntity)-[r2:HOTPOT_RELATION]-(t:HotpotEntity)
                            WHERE r1.graph_run_id=$gid AND r2.graph_run_id=$gid AND t <> s
                            RETURN [s, m, t] AS ns, [r1, r2] AS rs
                            ORDER BY coalesce(r1.confidence, 0.0) + coalesce(r2.confidence, 0.0) DESC
                            LIMIT $limit
                        """
                    result = session.run(cypher, skey=skey, gid=self.graph_run_id, limit=max_nodes_per_hop)
                    for rec in result:
                        ns = [dict(n) for n in rec["ns"]]
                        rs = [dict(r) for r in rec["rs"]]
                        path_edges = [_edge_from_props(r) for r in rs if r.get("source_chunk_id")]
                        if not path_edges:
                            continue
                        node_names = [str(n.get("name") or n.get("display_name") or n.get("name_norm")) for n in ns]
                        paths.append({"path_id": f"neo4j_path_{len(paths):06d}", "nodes": node_names, "edges": path_edges, "path_score": 0.0})
                        edges.extend(path_edges)
                        entities.extend(node_names)
        ranked = rank_paths(question, paths, top_k=top_k_paths)
        chunk_ids: list[str] = []
        for path in ranked:
            for edge in path.get("edges", []):
                cid = edge.get("source_chunk_id")
                if cid and cid not in chunk_ids:
                    chunk_ids.append(cid)
        chunks = self.get_chunks(chunk_ids[: max(10, top_k_paths * 2)])
        return {
            "seed_entities": [s.get("name") or s.get("node_id") for s in seeds],
            "retrieved_entities": list(dict.fromkeys([e for e in entities if e])),
            "retrieved_edges": edges,
            "retrieved_paths": ranked,
            "graph_evidence_chunks": chunks,
        }

    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch chunks by id."""
        if not chunk_ids:
            return []
        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $ids AS cid
                MATCH (c:HotpotChunk {id: cid})
                WHERE c.graph_run_id = $gid
                RETURN c
                """,
                ids=chunk_ids,
                gid=self.graph_run_id,
            )
            by_id = {dict(r["c"]).get("id"): dict(r["c"]) for r in result}
        out = []
        for idx, cid in enumerate(chunk_ids, start=1):
            props = by_id.get(cid)
            if props:
                out.append(_chunk_from_props(props, 1.0 / idx, idx, "graph_provenance_neo4j"))
        return out


class Neo4jTextRetriever:
    """Adapter exposing retrieve() for TextRAG."""

    def __init__(self, store: Neo4jHotpotQAStore):
        self.store = store

    def retrieve(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.store.text_search(question, top_k)


class Neo4jVectorRetriever:
    """Adapter exposing retrieve() for VectorRAG fallback."""

    def __init__(self, store: Neo4jHotpotQAStore):
        self.store = store

    def retrieve(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.store.vector_search(question, top_k)


class Neo4jGraphRetriever:
    """Adapter exposing retrieve() for GraphRAG."""

    def __init__(self, store: Neo4jHotpotQAStore):
        self.store = store

    def retrieve(self, question: str, depth: int = 1, top_k_paths: int = 5, max_nodes_per_hop: int = 10) -> dict[str, Any]:
        return self.store.graph_retrieve(question, depth, top_k_paths, max_nodes_per_hop)


def _chunk_from_props(props: dict[str, Any], score: float, rank: int, source: str) -> dict[str, Any]:
    return {
        "chunk_id": props.get("chunk_id") or props.get("id"),
        "doc_id": props.get("doc_id") or props.get("source_document_id"),
        "title": props.get("title"),
        "text": props.get("text"),
        "sentences": props.get("sentences") or [],
        "sentence_ids": props.get("sentence_ids") or [],
        "score": score,
        "rank": rank,
        "source": source,
    }


def _edge_from_props(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "head": props.get("head") or props.get("source"),
        "relation": props.get("type_norm") or props.get("type") or "related_to",
        "tail": props.get("tail") or props.get("target"),
        "source_chunk_id": props.get("source_chunk_id"),
        "source_title": props.get("source_title"),
        "source_sentence_ids": props.get("source_sentence_ids") or [],
        "supporting_quote": props.get("supporting_quote"),
        "confidence": props.get("confidence"),
    }
