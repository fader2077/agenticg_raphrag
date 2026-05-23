"""Local NetworkX/JSON graph store used when Neo4j is unavailable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx

from src.graph.canonicalization import entity_key
from src.io_utils import read_json, write_json


class LocalGraphStore:
    """A provenance-preserving graph store backed by NetworkX and JSON."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.chunk_map: dict[str, dict[str, Any]] = {}

    def add_chunk(self, chunk: dict[str, Any]) -> None:
        """Register a chunk for provenance lookup."""
        self.chunk_map[chunk["chunk_id"]] = chunk

    def add_edge(self, edge: dict[str, Any]) -> bool:
        """Add an edge only when provenance is complete enough for evaluation."""
        if not edge.get("source_chunk_id") or not edge.get("supporting_quote"):
            return False
        head_key = entity_key(edge["head"])
        tail_key = entity_key(edge["tail"])
        if not head_key or not tail_key or head_key == tail_key:
            return False
        self.graph.add_node(head_key, label=edge["head"])
        self.graph.add_node(tail_key, label=edge["tail"])
        edge_id = f"{head_key}::{edge['relation']}::{tail_key}::{edge['source_chunk_id']}"
        self.graph.add_edge(head_key, tail_key, key=edge_id, **edge, edge_id=edge_id)
        return True

    def entity_labels(self) -> list[str]:
        """Return graph entity labels."""
        return [data.get("label", node) for node, data in self.graph.nodes(data=True)]

    def save(self, path: Path) -> None:
        """Persist graph nodes, edges, chunks, and audit metrics."""
        path.mkdir(parents=True, exist_ok=True)
        nodes = [{"key": node, **data} for node, data in self.graph.nodes(data=True)]
        edges = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edges.append({"u": u, "v": v, "key": key, **data})
        write_json(path / "nodes.json", nodes)
        write_json(path / "edges.json", edges)
        write_json(path / "chunks.json", list(self.chunk_map.values()))
        write_json(path / "graph_metrics.json", self.audit())

    @classmethod
    def load(cls, path: Path) -> "LocalGraphStore":
        """Load a graph store from JSON artifacts."""
        store = cls()
        for chunk in read_json(path / "chunks.json", []):
            store.add_chunk(chunk)
        for node in read_json(path / "nodes.json", []):
            key = node.pop("key")
            store.graph.add_node(key, **node)
        for edge in read_json(path / "edges.json", []):
            u = edge.pop("u")
            v = edge.pop("v")
            key = edge.pop("key")
            store.graph.add_edge(u, v, key=key, **edge)
        return store

    def audit(self) -> dict[str, Any]:
        """Return basic graph quality metrics."""
        edges = list(self.graph.edges(keys=True, data=True))
        relation_count = len(edges)
        with_prov = sum(1 for _, _, _, data in edges if data.get("source_chunk_id") and data.get("supporting_quote"))
        degrees = [deg for _, deg in self.graph.degree()]
        super_node_ratio = 0.0
        if degrees:
            threshold = max(10, sorted(degrees)[int(0.95 * (len(degrees) - 1))])
            super_node_ratio = sum(1 for d in degrees if d >= threshold) / len(degrees)
        return {
            "entity_count": self.graph.number_of_nodes(),
            "relation_count": relation_count,
            "edge_provenance_coverage": with_prov / relation_count if relation_count else 0.0,
            "super_node_ratio": super_node_ratio,
            "backend": "networkx_json",
        }
