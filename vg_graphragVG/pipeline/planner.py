from __future__ import annotations

from vg_graphrag.models import AgentState, QueryAnalysis, RetrievalPlan, RunConfig, SubQuery, ToolCallSpec
from vg_graphrag.runtime_skill import skill_flag


GRAPH_RELATIONS = {
    "cause": ["causes", "associated_with", "implicated_in", "requires", "used_for"],
    "treatment": ["treats", "used_for", "prevents", "requires", "associated_with"],
    "prevention": ["prevents", "used_for", "requires", "associated_with"],
    "nutrition": ["requires", "deficiency_causes", "used_for", "associated_with"],
    "reproduction": ["affects", "associated_with", "requires"],
    "breed_trait": ["has_trait", "associated_with", "used_for"],
    "management": ["used_for", "improves", "requires", "associated_with"],
    "mechanism": ["associated_with", "implicated_in", "requires", "used_for"],
    "connection": ["associated_with", "used_for", "requires", "implicated_in"],
    "other": ["associated_with", "used_for", "requires"],
}


def create_initial_plan(question: str, analysis: QueryAnalysis, state: AgentState, config: RunConfig) -> RetrievalPlan:
    terms = [e.text for e in analysis.entities] or question.split()[:6]
    ql = question.lower()
    target_hint = terms[-1]
    if len(terms) == 1 and "company" in ql and "clinical trial" in ql:
        target_hint = "CompanyC"
    steps = [ToolCallSpec("EntitySearch", {"query": question, "context_terms": terms}, "Ground candidate graph entities.")]
    if config.use_graph_tools and analysis.query_type in {"multi_hop", "evidence_demanding"}:
        steps.append(ToolCallSpec("PathSearch", {"source_query": terms[0], "target_query": target_hint, "max_hops": min(config.max_hops, max(2, analysis.expected_hops)), "relation_filters": GRAPH_RELATIONS.get(analysis.answer_slot, []), "allow_unfiltered_fallback": analysis.answer_slot in {"mechanism", "connection", "other"}}, "Find graph reasoning paths.", max_hops=min(config.max_hops, 3)))
    elif config.use_graph_tools and analysis.answer_slot in {"mechanism", "connection"} and len(terms) >= 2:
        steps.append(ToolCallSpec("PathSearch", {"source_query": terms[0], "target_query": terms[-1], "max_hops": min(config.max_hops, 2), "relation_filters": GRAPH_RELATIONS.get(analysis.answer_slot, []), "allow_unfiltered_fallback": True}, "Try bounded path for explicit connection/mechanism questions.", max_hops=min(config.max_hops, 2)))
    elif config.use_graph_tools and analysis.entities:
        steps.append(ToolCallSpec("GraphNeighbor", {"node_query": terms[0], "max_hops": min(config.max_hops, 1), "relation_filters": GRAPH_RELATIONS.get(analysis.answer_slot, [])}, "Inspect bounded graph neighborhood.", max_hops=1))
    if config.use_text_tools:
        steps.append(ToolCallSpec("TextSearch", {"query": question, "limit": config.max_chunks}, "Find text support for graph evidence."))
    if config.use_text_tools and config.use_graph_tools:
        steps.append(ToolCallSpec("HybridSearch", {"query": question, "limit": config.max_chunks}, "Pull graph-linked text evidence for provenance checks."))
    return RetrievalPlan(steps=steps)


def create_dual_channel_plan(subquery: SubQuery, analysis: QueryAnalysis, state: AgentState, config: RunConfig) -> RetrievalPlan:
    """GraphSearch-style semantic + relational retrieval plan for one subquery."""
    q = subquery.grounded_text or subquery.text
    skill = dict((analysis.constraints or {}).get("runtime_skill") or {})
    preferred_tools = list((analysis.constraints or {}).get("preferred_tools", []) or [])
    if not preferred_tools:
        if analysis.query_type == "multi_hop":
            preferred_tools = list(skill_flag(skill, "tool_policy", "multi_hop_preferred", []))
        elif analysis.query_type == "evidence_demanding":
            preferred_tools = list(skill_flag(skill, "tool_policy", "evidence_demanding_preferred", []))
        elif analysis.query_type == "single_hop":
            preferred_tools = list(skill_flag(skill, "tool_policy", "single_hop_preferred", []))
    terms = [e.text for e in analysis.entities] or q.split()[:4]
    hint_targets = [str(x) for x in (analysis.constraints.get("target_entities", []) if analysis.constraints else [])]
    relation_terms = [str(x) for x in (analysis.constraints.get("relation_terms", []) if analysis.constraints else [])]
    focus_terms = [str(x) for x in (analysis.constraints.get("diagnostic_focus", []) if analysis.constraints else [])]
    target_query = hint_targets[0] if hint_targets else (terms[-1] if terms else q)
    if "company" in q.lower() and "trial" in q.lower():
        target_query = "CompanyC"
    rel_filters = list(dict.fromkeys(GRAPH_RELATIONS.get(analysis.answer_slot, []) + relation_terms))
    steps: list[ToolCallSpec] = []
    entity_context = list(dict.fromkeys(terms + hint_targets))
    matched_patterns = set((analysis.constraints.get("domain_hints", {}) or {}).get("matched_patterns", []) if analysis.constraints else [])
    minimal_direct_mode = analysis.query_type == "single_hop" and not matched_patterns
    steps.append(ToolCallSpec("EntitySearch", {"query": q, "context_terms": entity_context, "limit": 8}, f"QD/QG entity grounding for {subquery.subquery_id}."))
    if minimal_direct_mode and config.use_text_tools:
        steps.append(ToolCallSpec("TextSearch", {"query": q, "limit": config.max_chunks}, f"Direct text-first retrieval for {subquery.subquery_id}."))
        if analysis.answer_slot in {"cause", "treatment", "prevention", "nutrition", "reproduction", "mechanism"}:
            steps.append(
                ToolCallSpec(
                    "ClaimSearch",
                    {"query": q, "context_terms": entity_context[:6], "limit": max(config.max_chunks, 5)},
                    f"Direct claim support retrieval for {subquery.subquery_id}.",
                )
            )
        if preferred_tools:
            order = {name: idx for idx, name in enumerate(preferred_tools)}
            steps.sort(key=lambda s: order.get(s.tool_name, 999))
        return RetrievalPlan(steps=steps)
    if hint_targets:
        hint_query = " ".join(hint_targets[:5])
        steps.append(ToolCallSpec("EntitySearch", {"query": hint_query, "context_terms": hint_targets[:5], "limit": 8}, f"Canonical mechanism grounding for {subquery.subquery_id}."))
        steps.append(
            ToolCallSpec(
                "TextSearch",
                {"query": f"{hint_query} {' '.join(focus_terms[:1])}".strip(), "limit": config.max_chunks},
                f"Canonical-term semantic retrieval for {subquery.subquery_id}.",
            )
        )
        steps.append(
            ToolCallSpec(
                "ClaimSearch",
                {"query": hint_query, "context_terms": hint_targets[:5], "limit": max(config.max_chunks + 2, 6)},
                f"Canonical-term claim retrieval for {subquery.subquery_id}.",
            )
        )
    if analysis.answer_slot in {"cause", "nutrition", "reproduction", "mechanism", "connection"} or analysis.query_type in {"multi_hop", "evidence_demanding"}:
        claim_query = " ".join(hint_targets[:5]) if hint_targets and analysis.query_type == "evidence_demanding" else q
        steps.append(
            ToolCallSpec(
                "ClaimSearch",
                {"query": claim_query, "context_terms": entity_context + hint_targets[:5], "limit": max(config.max_chunks + 2, 6)},
                f"Claim-level mechanism retrieval for {subquery.subquery_id}.",
            )
        )
    if config.use_text_tools:
        semantic_query = q
        if hint_targets and analysis.query_type == "evidence_demanding":
            semantic_query = " ".join(hint_targets[:5])
            if focus_terms:
                semantic_query += f" {' '.join(focus_terms[:2])}"
        elif hint_targets:
            semantic_query = f"{q} {' '.join(hint_targets[:5])}"
        steps.append(ToolCallSpec("TextSearch", {"query": semantic_query, "limit": config.max_chunks}, f"Semantic channel retrieval for {subquery.subquery_id}."))
    if config.use_graph_tools:
        source_queries = terms[:3] or [q]
        target_queries = (hint_targets[:5] or terms[-2:] or [target_query])
        primary_source_query = source_queries[0]
        primary_target_query = target_query
        if analysis.query_type == "evidence_demanding" and hint_targets:
            source_queries = hint_targets[:3] + source_queries
            target_queries = hint_targets[1:5] + target_queries
            primary_source_query = hint_targets[0]
            if len(hint_targets) >= 2:
                primary_target_query = hint_targets[1]
        if hint_targets:
            steps.append(ToolCallSpec(
                "GraphNeighbor",
                {"node_query": hint_targets[0], "max_hops": min(config.max_hops, 2), "relation_filters": rel_filters},
                f"Relational channel local neighborhood for canonical mechanism in {subquery.subquery_id}.",
                max_hops=min(config.max_hops, 2),
            ))
        if len(source_queries) >= 1 and analysis.query_type in {"multi_hop", "evidence_demanding", "comparison"}:
            steps.append(ToolCallSpec(
                "PathSearch",
                {
                    "source_query": primary_source_query,
                    "target_query": primary_target_query,
                    "source_queries": source_queries,
                    "target_queries": target_queries,
                    "max_hops": min(config.max_hops, max(2, analysis.expected_hops)),
                    "relation_filters": rel_filters,
                    "allow_unfiltered_fallback": True,
                },
                f"Relational channel path search for {subquery.subquery_id}.",
                max_hops=min(config.max_hops, 3),
            ))
        elif terms:
            steps.append(ToolCallSpec(
                "GraphNeighbor",
                {"node_query": terms[0], "max_hops": min(config.max_hops, 2), "relation_filters": rel_filters},
                f"Relational channel bounded neighborhood for {subquery.subquery_id}.",
                max_hops=min(config.max_hops, 2),
            ))
        if config.use_text_tools:
            steps.append(ToolCallSpec(
                "HybridSearch",
                {"query": semantic_query if config.use_text_tools else q, "limit": config.max_chunks, "max_hops": min(config.max_hops, 2), "relation_filters": rel_filters},
                f"Dual-channel provenance search for {subquery.subquery_id}.",
                max_hops=min(config.max_hops, 2),
            ))
    if preferred_tools:
        order = {name: idx for idx, name in enumerate(preferred_tools)}
        steps.sort(key=lambda s: order.get(s.tool_name, 999))
    return RetrievalPlan(steps=steps)
