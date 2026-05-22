from __future__ import annotations

from vg_graphrag.models import AgentState, ToolResult, to_dict
from vg_graphrag.stores.graph_store import GraphStore


class GraphNeighbor:
    name = "GraphNeighbor"

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run(self, input: dict, state: AgentState) -> ToolResult:
        node_ids = input.get("node_ids") or []
        if isinstance(node_ids, str):
            node_ids = [node_ids]
        max_hops = int(input.get("max_hops", 1))
        relation_filters = input.get("relation_filters") or []
        results = []
        for node_id in node_ids:
            nb = self.graph.neighbors(node_id, max_hops=max_hops, relation_filters=relation_filters)
            results.append({
                "node_id": node_id,
                "nodes": [to_dict(n) for n in nb["nodes"]],
                "edges": [to_dict(e) for e in nb["edges"]],
            })
        return ToolResult(
            self.name,
            input,
            results=results,
            provenance_metadata={
                "graph_run_id": self.graph.graph_run_id,
                "graph_relation_type": getattr(self.graph, "relation_type", "RELATION"),
            },
        )
