from __future__ import annotations

from vg_graphrag.models import AgentState, ToolResult, to_dict
from vg_graphrag.stores.graph_store import GraphStore


class PathSearch:
    name = "PathSearch"

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def run(self, input: dict, state: AgentState) -> ToolResult:
        source_id = input.get("source_id")
        target_id = input.get("target_id")
        source_ids = input.get("source_ids") or ([] if not source_id else [source_id])
        target_ids = input.get("target_ids") or ([] if not target_id else [target_id])
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        if isinstance(target_ids, str):
            target_ids = [target_ids]
        if not source_ids or not target_ids:
            return ToolResult(self.name, input, errors=["source_id and target_id are required"])
        max_hops = int(input.get("max_hops", 3))
        rel_filters = input.get("relation_filters") or []
        allow_unfiltered_fallback = bool(input.get("allow_unfiltered_fallback", False))
        paths = []
        attempted_pairs = 0
        for sid in source_ids[:3]:
            for tid in target_ids[:3]:
                if sid == tid:
                    continue
                attempted_pairs += 1
                got = self.graph.paths(sid, tid, max_hops=max_hops, relation_filters=rel_filters)
                if got:
                    paths.extend(got)
        fallback_used = False
        if not paths and rel_filters and allow_unfiltered_fallback:
            # Path hit diagnostics: strict relation filters may over-prune; retry once without filters.
            for sid in source_ids[:3]:
                for tid in target_ids[:3]:
                    if sid == tid:
                        continue
                    got = self.graph.paths(sid, tid, max_hops=max_hops, relation_filters=[])
                    if got:
                        paths.extend(got)
            fallback_used = bool(paths)
        # de-dup by node-chain signature
        uniq = {}
        for p in paths:
            sig = "->".join(p.get("nodes") or [])
            if sig and sig not in uniq:
                uniq[sig] = p
        results = [{"nodes": p["nodes"], "edges": [to_dict(e) for e in p["edges"]]} for p in uniq.values()]
        relation_type_fallback_used = any(
            bool((e.get("provenance") or {}).get("relation_type_fallback_used"))
            for p in results
            for e in p.get("edges", [])
            if isinstance(e, dict)
        )
        return ToolResult(
            self.name,
            input,
            results=results,
            provenance_metadata={
                "graph_run_id": self.graph.graph_run_id,
                "graph_relation_type": getattr(self.graph, "relation_type", "RELATION"),
                "relation_type_fallback_used": relation_type_fallback_used,
                "path_hit": bool(paths),
                "fallback_unfiltered_used": fallback_used,
                "relation_filters_applied": rel_filters,
                "allow_unfiltered_fallback": allow_unfiltered_fallback,
                "attempted_pairs": attempted_pairs,
            },
        )
