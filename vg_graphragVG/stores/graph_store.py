from __future__ import annotations

from typing import List, Protocol

from vg_graphrag.models import Edge, Node


class GraphStore(Protocol):
    graph_run_id: str

    def search_entities(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]: ...
    def search_claims(self, query: str, context_terms: list[str] | None = None, limit: int = 10) -> List[dict]: ...

    def neighbors(self, node_id: str, max_hops: int = 1, relation_filters: list[str] | None = None) -> dict: ...

    def paths(self, source_id: str, target_id: str, max_hops: int = 3, relation_filters: list[str] | None = None) -> List[dict]: ...

    def get_node(self, node_id: str) -> Node | None: ...

    def get_edge(self, edge_id: str) -> Edge | None: ...
