from __future__ import annotations

import re

from vg_graphrag.models import ChannelEvidence, LogicDraft, QueryAnalysis, RunConfig, ToolCallSpec, VerifierReport
from vg_graphrag.runtime_skill import skill_flag


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "what", "which", "how", "does", "from", "into", "under"}
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 2 and t.lower() not in stop}


def _evidence_focus_scores(ev: ChannelEvidence, analysis: QueryAnalysis) -> tuple[float, float]:
    focus_terms = set()
    for term in analysis.constraints.get("canonical_terms", []) if analysis.constraints else []:
        focus_terms |= _tokens(str(term).replace("_", " "))
    for term in analysis.constraints.get("alias_terms", []) if analysis.constraints else []:
        focus_terms |= _tokens(str(term).replace("_", " "))
    if not focus_terms:
        return 1.0, 1.0
    semantic_best = 0.0
    for item in ev.semantic_evidence:
        txt = item.get("text") or (item.get("chunk") or {}).get("text", "")
        overlap = len(focus_terms & _tokens(txt))
        semantic_best = max(semantic_best, float(overlap))
    relational_best = 0.0
    for item in ev.relational_evidence:
        rel_text_parts = []
        if item.get("claim"):
            claim = item.get("claim") or {}
            rel_text_parts.extend([
                str(claim.get("claim_text", "")),
                str(claim.get("head", "")),
                str(claim.get("relation", "")),
                str(claim.get("tail", "")),
                str(claim.get("supporting_quote", "")),
            ])
        for edge in item.get("edges", []) or []:
            rel_text_parts.extend([str(edge.get("source", "")), str(edge.get("relation", "")), str(edge.get("target", ""))])
        for node in item.get("nodes", []) or []:
            if isinstance(node, dict):
                rel_text_parts.append(str(node.get("name") or node.get("node_id") or ""))
            else:
                rel_text_parts.append(str(node))
        overlap = len(focus_terms & _tokens(" ".join(rel_text_parts)))
        relational_best = max(relational_best, float(overlap))
    return semantic_best, relational_best


def _is_qe_subquery(ev: ChannelEvidence) -> bool:
    return "_qe_" in ev.subquery_id.lower()


def _has_claim_evidence(ev: ChannelEvidence) -> bool:
    return any(bool(item.get("claim")) for item in ev.relational_evidence) or any(bool(item.get("claim")) for item in ev.semantic_evidence)


def verify_graphsearch_evidence(
    question: str,
    analysis: QueryAnalysis,
    evidence_items: list[ChannelEvidence],
    draft: LogicDraft,
    iteration: int,
    config: RunConfig,
) -> VerifierReport:
    """EV: verify per-subquery semantic/relational evidence sufficiency."""
    missing_map: dict[str, list[str]] = {}
    failures: list[str] = []
    focus_hits = 0
    complete_hits = 0
    core_items = [ev for ev in evidence_items if not _is_qe_subquery(ev)] or evidence_items
    core_focus_hits = 0
    core_complete_hits = 0
    for ev in evidence_items:
        missing = list(ev.missing)
        semantic_focus, relational_focus = _evidence_focus_scores(ev, analysis)
        if analysis.query_type in {"multi_hop", "evidence_demanding"} and not ev.relational_evidence:
            if "relational_graph_evidence" not in missing:
                missing.append("relational_graph_evidence")
        need_claim_evidence = analysis.answer_slot in {"cause", "nutrition", "reproduction", "mechanism", "connection"}
        has_relational_path = bool(ev.relational_evidence)
        has_semantic_text = bool(ev.semantic_evidence)
        if need_claim_evidence and not _has_claim_evidence(ev) and not (has_relational_path and has_semantic_text):
            if "mechanism_claim_evidence" not in missing:
                missing.append("mechanism_claim_evidence")
        if not ev.semantic_evidence:
            if "semantic_text_evidence" not in missing:
                missing.append("semantic_text_evidence")
        if analysis.constraints.get("canonical_terms"):
            if semantic_focus <= 0 and relational_focus <= 0:
                if "focus_aligned_evidence" not in missing:
                    missing.append("focus_aligned_evidence")
            if semantic_focus > 0 or relational_focus > 0:
                focus_hits += 1
            if semantic_focus > 0 and (relational_focus > 0 or analysis.query_type not in {"multi_hop", "evidence_demanding"}):
                complete_hits += 1
        if ev in core_items and (semantic_focus > 0 or relational_focus > 0):
            core_focus_hits += 1
        if ev in core_items and semantic_focus > 0 and (relational_focus > 0 or analysis.query_type not in {"multi_hop", "evidence_demanding"}):
            core_complete_hits += 1
        if missing and (ev in core_items or core_complete_hits == 0):
            missing_map[ev.subquery_id] = missing
            failures.extend(missing)
    if not evidence_items:
        missing_map["SQ1"] = ["subquery_evidence"]
        failures.append("subquery_evidence")
    if draft.evidence_gaps:
        for sid, gaps in draft.evidence_gaps.items():
            missing_map.setdefault(sid, [])
            for gap in gaps:
                if gap not in missing_map[sid]:
                    missing_map[sid].append(gap)

    total_slots = max(1, len(evidence_items) * 2)
    present = sum(1 for ev in evidence_items if ev.semantic_evidence) + sum(1 for ev in evidence_items if ev.relational_evidence)
    suff = present / total_slots
    if analysis.query_type == "single_hop" and any(ev.semantic_evidence for ev in evidence_items):
        suff = max(suff, 0.6)
    if evidence_items and analysis.constraints.get("canonical_terms"):
        suff += 0.15 * (core_focus_hits / max(1, len(core_items)))
        suff = min(suff, 1.0)

    actions: list[ToolCallSpec] = []
    for sid, missing in missing_map.items():
        ev = next((x for x in evidence_items if x.subquery_id == sid), None)
        q = ev.grounded_query if ev else question
        if "semantic_text_evidence" in missing:
            actions.append(ToolCallSpec("TextSearch", {"query": q, "limit": config.max_chunks}, f"QE semantic repair for {sid}."))
        if "relational_graph_evidence" in missing:
            actions.append(ToolCallSpec("HybridSearch", {"query": q, "limit": config.max_chunks, "max_hops": min(config.max_hops, 2)}, f"QE relational repair for {sid}."))
        if "mechanism_claim_evidence" in missing:
            hint_terms = [str(x) for x in (analysis.constraints.get("canonical_terms", []) if analysis.constraints else [])[:5]]
            actions.append(ToolCallSpec("ClaimSearch", {"query": q, "context_terms": hint_terms, "limit": max(config.max_chunks + 2, 6)}, f"QE claim repair for {sid}."))
        if "focus_aligned_evidence" in missing:
            hint_terms = [str(x) for x in (analysis.constraints.get("canonical_terms", []) if analysis.constraints else [])[:5]]
            focus_text = "; ".join(str(x) for x in (analysis.constraints.get("diagnostic_focus", []) if analysis.constraints else [])[:2])
            hint_query = q
            if hint_terms:
                hint_query += f" Focus on {', '.join(hint_terms)}."
            if focus_text:
                hint_query += f" Mechanism target: {focus_text}."
            actions.append(ToolCallSpec("TextSearch", {"query": hint_query, "limit": config.max_chunks}, f"QE focus repair for {sid}."))

    skill = dict((analysis.constraints or {}).get("runtime_skill") or {})
    complete_subqueries = sum(1 for ev in core_items if ev.semantic_evidence and ev.relational_evidence)
    hard_fail = analysis.query_type in {"multi_hop", "evidence_demanding"} and complete_subqueries == 0
    if analysis.constraints.get("canonical_terms") and core_focus_hits == 0:
        hard_fail = True
    if analysis.query_type == "evidence_demanding" and core_complete_hits == 0:
        hard_fail = True
    missing_flat = sorted({m for vals in missing_map.values() for m in vals})
    force_refine = False
    if skill_flag(skill, "reflection_policy", "force_refine_on_missing_focus", False) and "focus_aligned_evidence" in missing_flat:
        force_refine = True
    if skill_flag(skill, "reflection_policy", "force_refine_on_missing_claim", False) and "mechanism_claim_evidence" in missing_flat:
        force_refine = True
    if skill_flag(skill, "reflection_policy", "force_refine_on_missing_path_for_indirect", False):
        if analysis.query_type in {"multi_hop", "evidence_demanding"} and "relational_graph_evidence" in missing_flat:
            force_refine = True
    if analysis.query_type in {"multi_hop", "evidence_demanding"} and iteration == 0 and core_complete_hits < max(1, len(core_items)):
        force_refine = True
    if analysis.query_type == "evidence_demanding" and iteration == 0 and analysis.constraints.get("canonical_terms"):
        force_refine = True
    core_semantic_hits = sum(1 for ev in core_items if ev.semantic_evidence)
    strong_semantic_accept = (
        analysis.query_type == "evidence_demanding"
        and core_focus_hits > 0
        and core_semantic_hits > 0
        and suff >= max(config.refine_threshold, 0.5)
    )
    if suff >= config.accept_threshold and not hard_fail:
        verdict = "accept"
    elif suff >= config.accept_threshold and not missing_map and not actions:
        verdict = "accept"
    elif strong_semantic_accept and iteration + 1 >= config.max_iterations:
        verdict = "accept"
    elif iteration + 1 >= config.max_iterations:
        verdict = "abstain" if suff < config.refine_threshold or hard_fail else "accept"
    else:
        verdict = "refine"
    if force_refine and iteration + 1 < config.max_iterations:
        verdict = "refine"
    return VerifierReport(
        verdict=verdict,
        sufficiency_score=suff,
        failure_modes=sorted(set(failures)),
        missing_information=missing_flat,
        recommended_actions=actions,
        rationale=f"graphsearch_sufficiency={suff:.3f}; missing_subqueries={len(missing_map)}",
        missing_evidence_map=missing_map,
    )
