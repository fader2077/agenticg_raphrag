"""GraphRAG retrieval wrapper for hop-1 and hop-2 settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.graph.entity_extraction import link_question_entities
from src.graph.graph_expand import expand_graph
from src.graph.graph_store import LocalGraphStore
from src.graph.path_rank import rank_paths


class GraphRAGRetriever:
    """Graph retriever using explicit entity linking, expansion, and path ranking."""

    def __init__(self, index_dir: Path):
        self.store = LocalGraphStore.load(index_dir)

    def retrieve(
        self,
        question: str,
        depth: int = 1,
        top_k_paths: int = 5,
        max_nodes_per_hop: int = 10,
    ) -> dict[str, Any]:
        """Retrieve graph entities, edges, paths, and provenance chunks."""
        seeds = link_question_entities(question, self.store.entity_labels())
        expanded = expand_graph(self.store, seeds, depth=depth, max_nodes_per_hop=max_nodes_per_hop)
        paths = rank_paths(question, expanded["paths"], top_k=top_k_paths)
        chunk_ids: list[str] = []
        for path in paths:
            for edge in path.get("edges", []):
                cid = edge.get("source_chunk_id")
                if cid and cid not in chunk_ids:
                    chunk_ids.append(cid)
        if not chunk_ids:
            for edge in expanded["edges"]:
                cid = edge.get("source_chunk_id")
                if cid and cid not in chunk_ids:
                    chunk_ids.append(cid)
        chunks = []
        for idx, cid in enumerate(chunk_ids[: max(10, top_k_paths * 2)], start=1):
            chunk = dict(self.store.chunk_map.get(cid, {}))
            if chunk:
                chunk["score"] = 1.0 / idx
                chunk["source"] = "graph_provenance"
                chunks.append(chunk)
        return {
            "seed_entities": seeds,
            "retrieved_entities": expanded["entities"],
            "retrieved_edges": expanded["edges"],
            "retrieved_paths": paths,
            "graph_evidence_chunks": chunks,
        }
