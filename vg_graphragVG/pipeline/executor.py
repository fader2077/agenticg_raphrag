from __future__ import annotations

from typing import Dict

from vg_graphrag.models import AgentState, RetrievalPlan, RunConfig, ToolResult, to_dict

_ENTITY_STOP = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "from", "as", "at", "it",
    "that", "this", "what", "which", "how", "when", "where", "why",
}


def _is_low_quality_entity_id(node_id: str) -> bool:
    x = (node_id or "").strip().lower()
    if not x or x in _ENTITY_STOP:
        return True
    if len(x) <= 1:
        return True
    if len(x) <= 2 and x.isalpha():
        return True
    return False


def _best_entity(state: AgentState, query: str) -> str | None:
    query_l = (query or "").lower()
    for tr in reversed(state.tool_history):
        if tr.tool_name != "EntitySearch":
            continue
        for r in tr.results:
            if r.get("name", "").lower() == query_l or query_l in r.get("name", "").lower():
                nid = r.get("node_id")
                return None if _is_low_quality_entity_id(str(nid)) else nid
    for tr in reversed(state.tool_history):
        if tr.tool_name != "EntitySearch":
            continue
        if tr.results:
            nid = tr.results[0].get("node_id")
            return None if _is_low_quality_entity_id(str(nid)) else nid
    return None


def _best_entity_candidates(state: AgentState, query: str, limit: int = 3) -> list[str]:
    query_l = (query or "").lower()
    out: list[tuple[float, str]] = []
    for tr in reversed(state.tool_history):
        if tr.tool_name != "EntitySearch":
            continue
        for r in tr.results:
            nid = r.get("node_id")
            if not nid:
                continue
            name_l = str(r.get("name", "")).lower()
            ms = float(r.get("match_score", 0) or 0)
            boost = 2.0 if query_l and query_l in name_l else 0.0
            out.append((ms + boost, nid))
    out.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    ids = []
    for _, nid in out:
        if _is_low_quality_entity_id(str(nid)):
            continue
        if nid in seen:
            continue
        seen.add(nid)
        ids.append(nid)
        if len(ids) >= limit:
            break
    return ids


def _collect_entity_candidates(state: AgentState, queries: list[str], limit: int = 5) -> list[str]:
    seen = set()
    out: list[str] = []
    for query in queries:
        for nid in _best_entity_candidates(state, query, limit=limit):
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
            if len(out) >= limit:
                return out
    return out


def execute_plan(plan: RetrievalPlan, tools: Dict[str, object], state: AgentState, config: RunConfig) -> list[ToolResult]:
    out = []
    for step in plan.steps:
        if state.tool_calls >= config.max_tool_calls:
            break
        tool = tools.get(step.tool_name)
        if not tool:
            result = ToolResult(step.tool_name, step.input, errors=["tool_not_available"])
        else:
            inp = dict(step.input)
            if step.tool_name == "GraphNeighbor" and "node_ids" not in inp:
                node = _best_entity(state, inp.pop("node_query", ""))
                inp["node_ids"] = [node] if node else []
            if step.tool_name == "PathSearch" and ("source_id" not in inp or "target_id" not in inp):
                source_query = inp.pop("source_query", "")
                target_query = inp.pop("target_query", "")
                source_queries = inp.pop("source_queries", [])
                target_queries = inp.pop("target_queries", [])
                source = _best_entity(state, source_query)
                target = _best_entity(state, target_query)
                inp["source_id"], inp["target_id"] = source, target
                inp["source_ids"] = _collect_entity_candidates(state, [source_query] + list(source_queries), limit=5) or ([source] if source else [])
                inp["target_ids"] = _collect_entity_candidates(state, [target_query] + list(target_queries), limit=5) or ([target] if target else [])
            if step.tool_name == "HybridSearch" and "node_ids" not in inp:
                inp["node_ids"] = _best_entity_candidates(state, inp.get("query", ""), limit=3)
            result = tool.run(inp, state)  # type: ignore[attr-defined]
        state.tool_calls += 1
        state.tool_history.append(result)
        state.tool_trace.append({
            "iteration": state.iterations,
            "tool_name": step.tool_name,
            "input": step.input,
            "rationale": step.rationale,
            "result_count": len(result.results),
            "errors": result.errors,
            "graph_relation_type": (result.provenance_metadata or {}).get("graph_relation_type"),
            "relation_type_fallback_used": bool((result.provenance_metadata or {}).get("relation_type_fallback_used")),
            "relation_type_result_count": len(result.results) if step.tool_name in {"PathSearch", "GraphNeighbor"} else 0,
        })
        out.append(result)
    return out
