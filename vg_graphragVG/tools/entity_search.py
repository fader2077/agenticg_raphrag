from __future__ import annotations

from vg_graphrag.models import AgentState, ToolResult
from vg_graphrag.stores.graph_store import GraphStore


class EntitySearch:
    name = "EntitySearch"

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run(self, input: dict, state: AgentState) -> ToolResult:
        query = input.get("query") or state.question
        context_terms = input.get("context_terms") or []
        results = self.graph.search_entities(query, context_terms=context_terms, limit=int(input.get("limit", 10)))
        return ToolResult(self.name, input, results=results, provenance_metadata={"graph_run_id": self.graph.graph_run_id})
