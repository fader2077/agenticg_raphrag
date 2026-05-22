from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

from vg_graphrag.adapters.graph_run_loader import load_graph_run
from vg_graphrag.demo.corpus import build_demo_stores
from vg_graphrag.models import AgentState, ClaimFamilyDecision, FinalAnswer, RetrievalPlan, RunConfig, SubQuery, ToolCallSpec, to_dict
from vg_graphrag.pipeline.analyzer import analyze_query
from vg_graphrag.pipeline.claim_family_arbitrator import arbitrate_claim_family
from vg_graphrag.pipeline.context_refinement import refine_context
from vg_graphrag.pipeline.decomposition import decompose_query, decompose_relational_queries
from vg_graphrag.pipeline.evidence_builder import build_evidence_package
from vg_graphrag.pipeline.executor import execute_plan
from vg_graphrag.pipeline.graphsearch_verifier import verify_graphsearch_evidence
from vg_graphrag.pipeline.grounding import ground_subquery
from vg_graphrag.pipeline.logic_drafting import draft_logic
from vg_graphrag.pipeline.planner import create_dual_channel_plan
from vg_graphrag.pipeline.query_expansion import expand_queries
from vg_graphrag.pipeline.refiner import refine_plan
from vg_graphrag.pipeline.self_reflection import build_self_reflection
from vg_graphrag.pipeline.synthesizer import synthesize_vg_native_answer
from vg_graphrag.pipeline.verifier import verify_evidence
from vg_graphrag.runtime_skill import load_runtime_skill
from vg_graphrag.runtime_skill import skill_flag
from vg_graphrag.tools import ClaimSearch, EntitySearch, GraphNeighbor, HybridSearch, PathSearch, TextSearch


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _merge_subqueries(existing, new_items):
    seen = {getattr(x, "subquery_id", None) for x in (existing or [])}
    out = list(existing or [])
    for item in new_items or []:
        sid = getattr(item, "subquery_id", None)
        if sid not in seen:
            seen.add(sid)
            out.append(item)
    return out


def _merge_query_guidance(analysis, guidance: dict | None):
    if not guidance:
        return analysis
    candidate_mechanisms = [str(x) for x in guidance.get("candidate_mechanisms", []) if str(x).strip()]
    evidence_terms = [str(x) for x in guidance.get("evidence_terms", []) if str(x).strip()]
    retrieval_focus = str(guidance.get("retrieval_focus", "") or "").strip()
    constraints = dict(analysis.constraints or {})
    constraints["llm_guidance"] = guidance
    existing_canonical = list(constraints.get("canonical_terms", []))
    if existing_canonical:
        preserved = []
        canon_l = " ".join(existing_canonical).lower()
        for mech in candidate_mechanisms:
            mech_l = mech.lower().replace("_", " ")
            if mech_l in canon_l or any(tok in canon_l for tok in mech_l.split()):
                preserved.append(mech)
        constraints["canonical_terms"] = _dedupe_keep_order(existing_canonical + preserved)
    else:
        constraints["canonical_terms"] = _dedupe_keep_order(existing_canonical + candidate_mechanisms)
    constraints["alias_terms"] = _dedupe_keep_order(list(constraints.get("alias_terms", [])) + evidence_terms + candidate_mechanisms)
    constraints["target_entities"] = _dedupe_keep_order(list(constraints.get("target_entities", [])) + evidence_terms + candidate_mechanisms)
    if retrieval_focus:
        constraints["diagnostic_focus"] = _dedupe_keep_order(list(constraints.get("diagnostic_focus", [])) + [retrieval_focus])
    analysis.constraints = constraints
    return analysis


def _merge_reflection(analysis, reflection):
    if reflection is None:
        return analysis
    constraints = dict(analysis.constraints or {})
    existing_canonical = list(constraints.get("canonical_terms", []))
    constraints["canonical_terms"] = _dedupe_keep_order(existing_canonical + [str(x) for x in reflection.updated_focus_terms if str(x).strip()])
    constraints["alias_terms"] = _dedupe_keep_order(list(constraints.get("alias_terms", [])) + list(reflection.updated_focus_terms))
    constraints["avoid_terms"] = _dedupe_keep_order(list(constraints.get("avoid_terms", [])) + list(reflection.avoid_terms))
    if reflection.preferred_tools:
        constraints["preferred_tools"] = _dedupe_keep_order(list(constraints.get("preferred_tools", [])) + list(reflection.preferred_tools))
    if reflection.improved_strategy:
        constraints["diagnostic_focus"] = _dedupe_keep_order(list(constraints.get("diagnostic_focus", [])) + [reflection.improved_strategy])
    analysis.constraints = constraints
    return analysis


def _tool_bank(graph, text) -> Dict[str, object]:
    return {
        "ClaimSearch": ClaimSearch(graph),
        "EntitySearch": EntitySearch(graph),
        "GraphNeighbor": GraphNeighbor(graph),
        "PathSearch": PathSearch(graph),
        "TextSearch": TextSearch(text),
        "HybridSearch": HybridSearch(graph, text),
    }


def _filter_plan_for_graph_reliance(plan: RetrievalPlan, mode: str) -> RetrievalPlan:
    if mode != "text_claim_only_agentic":
        return plan
    kept: list[ToolCallSpec] = []
    for step in plan.steps:
        if step.tool_name in {"GraphNeighbor", "PathSearch"}:
            continue
        kept.append(step)
    return RetrievalPlan(kept)


def _apply_graph_reliance_mode(evidence, state: AgentState, mode: str):
    if mode in {"text_claim_only_agentic", "graph_hint_only"}:
        removed = len(evidence.supporting_paths or [])
        if removed:
            state.tool_trace.append(
                {
                    "iteration": state.iterations,
                    "action": "graph_reliance_filter",
                    "graph_reliance_mode": mode,
                    "unsupported_graph_path_rejected_count": removed,
                    "kept_paths": 0,
                }
            )
        evidence.supporting_paths = []
        evidence.subgraph_edges = []
        evidence.coverage_flags["path_found"] = False
        evidence.score_components["path_completeness"] = 0.0
        evidence.score_components["relation_relevance"] = 0.0
        evidence.provenance_summary["path_count"] = 0
        evidence.provenance_summary["edge_count"] = 0
        evidence.evidence_score = (
            0.25 * evidence.score_components.get("entity_coverage", 0.0)
            + 0.25 * evidence.score_components.get("path_completeness", 0.0)
            + 0.20 * evidence.score_components.get("text_support", 0.0)
            + 0.15 * evidence.score_components.get("relation_relevance", 0.0)
            + 0.10 * evidence.score_components.get("source_reliability", 0.0)
            + 0.05 * evidence.score_components.get("recency", 0.0)
        )
        return evidence
    if mode == "graph_evidence_allowed":
        kept = []
        removed = 0
        for path in evidence.supporting_paths or []:
            if path.text_support_ids:
                kept.append(path)
            else:
                removed += 1
        evidence.supporting_paths = kept
        evidence.provenance_summary["path_count"] = len(kept)
        evidence.coverage_flags["path_found"] = bool(kept) or bool(evidence.subgraph_edges)
        if removed:
            state.tool_trace.append(
                {
                    "iteration": state.iterations,
                    "action": "graph_reliance_filter",
                    "graph_reliance_mode": mode,
                    "unsupported_graph_path_rejected_count": removed,
                    "kept_paths": len(kept),
                }
            )
    return evidence


def _packaging_slots_from_text(text: str) -> set[str]:
    t = (text or "").lower()
    slots = set()
    if any(k in t for k in ["if ", "when ", "despite", "under ", "during ", "in goats with", "in does with", "context"]):
        slots.add("condition")
    if any(k in t for k in ["because", "due to", "causes", "mechanism", "pathway", "leads to", "results in", "mediated"]):
        slots.add("mechanism")
    if any(k in t for k in ["therefore", "thus", "outcome", "result", "reduced", "improved", "decline", "increase"]):
        slots.add("outcome")
    if any(k in t for k in ["management", "prioritize", "intervention", "hygiene", "monitor", "feeding", "environment"]):
        slots.add("management")
    if any(k in t for k in ["prevention", "prevent", "biosecurity", "quarantine", "avoid"]):
        slots.add("prevention")
    if any(k in t for k in ["treatment", "treat", "therapy", "antibiotic", "drug", "medication"]):
        slots.add("treatment")
    return slots


def _apply_packaging_mode(
    question: str,
    analysis,
    evidence,
    family_decision: ClaimFamilyDecision | None,
    mode: str,
):
    input_count = len(evidence.supporting_claims or []) + len(evidence.supporting_chunks or []) + len(evidence.supporting_paths or [])
    trace = {
        "fully_auto_family_packaging_mode": mode,
        "mechanism_chain_slots_present": [],
        "missing_mechanism_slots": [],
        "graph_hint_bound_to_text_count": 0,
        "unbound_graph_hint_count": 0,
        "duplicate_evidence_dropped_count": 0,
        "packaging_warning": "",
        "answer_plan_generated": False,
        "packaging_input_evidence_count": input_count,
        "packaging_output_evidence_count": input_count,
    }
    if mode != "mechanism_chain_pack_v1":
        return evidence, trace

    packaged = deepcopy(evidence)
    dedup_dropped = 0

    claim_seen = set()
    dedup_claims = []
    for c in packaged.supporting_claims or []:
        key = str(c.get("claim_id") or "").strip() or " ".join(
            [
                str(c.get("head") or ""),
                str(c.get("relation") or ""),
                str(c.get("tail") or ""),
                str(c.get("claim_text") or ""),
            ]
        ).strip().lower()
        if key in claim_seen:
            dedup_dropped += 1
            continue
        claim_seen.add(key)
        dedup_claims.append(c)
    packaged.supporting_claims = dedup_claims

    chunk_seen = set()
    dedup_chunks = []
    for ch in packaged.supporting_chunks or []:
        key = str(getattr(ch, "chunk_id", "") or "").strip() or " ".join((getattr(ch, "text", "") or "").split())[:300].lower()
        if key in chunk_seen:
            dedup_dropped += 1
            continue
        chunk_seen.add(key)
        dedup_chunks.append(ch)
    packaged.supporting_chunks = dedup_chunks

    slot_claims: dict[str, list[dict]] = defaultdict(list)
    for c in packaged.supporting_claims:
        txt = " ".join(
            str(c.get(k) or "")
            for k in ("claim_text", "head", "relation", "tail", "supporting_quote")
        )
        slots = _packaging_slots_from_text(txt)
        if not slots:
            slot_claims["other"].append(c)
        else:
            for s in slots:
                slot_claims[s].append(c)

    order = ["condition", "mechanism", "outcome", "management", "prevention", "treatment", "other"]
    ordered_claims = []
    used = set()
    for slot in order:
        for c in slot_claims.get(slot, []):
            cid = str(c.get("claim_id") or id(c))
            if cid in used:
                continue
            used.add(cid)
            ordered_claims.append(c)
    if ordered_claims:
        packaged.supporting_claims = ordered_claims

    for p in packaged.supporting_paths or []:
        if p.text_support_ids:
            trace["graph_hint_bound_to_text_count"] += 1
        else:
            trace["unbound_graph_hint_count"] += 1

    present_slots = [s for s in order[:-1] if slot_claims.get(s)]
    trace["mechanism_chain_slots_present"] = present_slots
    trace["missing_mechanism_slots"] = [s for s in ["condition", "mechanism", "outcome"] if s not in present_slots]
    trace["duplicate_evidence_dropped_count"] = dedup_dropped
    if trace["missing_mechanism_slots"]:
        trace["packaging_warning"] = "missing_core_mechanism_slots"
    if family_decision and family_decision.selected_family:
        trace["answer_plan_generated"] = True
    out_count = len(packaged.supporting_claims or []) + len(packaged.supporting_chunks or []) + len(packaged.supporting_paths or [])
    trace["packaging_output_evidence_count"] = out_count
    return packaged, trace


def _evidence_slots_text(text: str) -> set[str]:
    t = (text or "").lower()
    slots = set()
    if any(k in t for k in ["if ", "when ", "despite", "during", "under ", "condition"]):
        slots.add("condition")
    if any(k in t for k in ["because", "causes", "leads to", "results in", "mechanism", "pathway"]):
        slots.add("mechanism")
    if any(k in t for k in ["therefore", "thus", "outcome", "result", "reduce", "increase", "decline"]):
        slots.add("outcome")
    if any(k in t for k in ["management", "prioritize", "intervention", "hygiene", "monitor", "feeding"]):
        slots.add("management")
    if any(k in t for k in ["prevention", "prevent", "biosecurity", "quarantine"]):
        slots.add("prevention")
    if any(k in t for k in ["treatment", "treat", "therapy", "antibiotic", "drug", "medication"]):
        slots.add("treatment")
    return slots


def _apply_family_hint_rerank_mode(
    evidence,
    family_decision: ClaimFamilyDecision | None,
    mode: str,
):
    before_ids = []
    for c in evidence.supporting_claims or []:
        before_ids.append(str(c.get("claim_id") or c.get("relation_id") or c.get("head") or "claim"))
    for ch in evidence.supporting_chunks or []:
        before_ids.append(str(getattr(ch, "chunk_id", None) or "chunk"))
    trace = {
        "fully_auto_family_hint_rerank_mode": mode,
        "family_hint_rerank_applied": False,
        "evidence_count_before_rerank": len(before_ids),
        "evidence_count_after_rerank": len(before_ids),
        "top_evidence_ids_before": before_ids[:10],
        "top_evidence_ids_after": before_ids[:10],
        "family_hint_score_distribution": [],
        "bound_graph_hint_count": 0,
        "unbound_graph_hint_count": 0,
        "rerank_warning": "",
        "exact_duplicate_dropped_count": 0,
    }
    if mode != "family_hint_rerank_v1":
        return evidence, trace
    if not family_decision:
        trace["rerank_warning"] = "no_family_decision"
        return evidence, trace

    sel = str(family_decision.selected_family or "")
    score_row = (family_decision.family_scores or {}).get(sel, {}) if sel else {}
    qa = float(score_row.get("query_alignment_score", score_row.get("question_focus_alignment", 0.0)) or 0.0)
    co = float(score_row.get("claim_overlap_score", 0.0) or 0.0)
    te = float(score_row.get("text_evidence_support_score", score_row.get("semantic_text_support", 0.0)) or 0.0)
    rel = float(score_row.get("graph_relation_support", score_row.get("graph_support_score", 0.0)) or 0.0)
    ent = float(score_row.get("entity_support_score", 0.0) or 0.0)

    dedup_claims = []
    seen_claim = set()
    dropped = 0
    for c in evidence.supporting_claims or []:
        key = str(c.get("claim_id") or c.get("relation_id") or "").strip() or " ".join(
            [str(c.get("head") or ""), str(c.get("relation") or ""), str(c.get("tail") or ""), str(c.get("claim_text") or "")]
        ).strip().lower()
        if key in seen_claim:
            dropped += 1
            continue
        seen_claim.add(key)
        dedup_claims.append(c)

    dedup_chunks = []
    seen_chunks = set()
    for ch in evidence.supporting_chunks or []:
        key = str(getattr(ch, "chunk_id", "") or "").strip() or " ".join((getattr(ch, "text", "") or "").split())[:300].lower()
        if key in seen_chunks:
            dropped += 1
            continue
        seen_chunks.add(key)
        dedup_chunks.append(ch)

    scored_claims = []
    for i, c in enumerate(dedup_claims):
        txt = " ".join(str(c.get(k) or "") for k in ("claim_text", "supporting_quote", "head", "relation", "tail"))
        slots = _evidence_slots_text(txt)
        slot_bonus = 0.0
        if any(s in slots for s in {"condition", "mechanism", "outcome"}):
            slot_bonus += 0.15
        if any(s in slots for s in {"management", "prevention", "treatment"}):
            slot_bonus += 0.08
        prov_bonus = 0.05 if str(c.get("source_chunk_id") or c.get("chunk_id") or "").strip() else 0.0
        family_hint_score = 0.30 * qa + 0.25 * co + 0.25 * te + 0.10 * rel + 0.05 * ent + slot_bonus + prov_bonus
        orig = 1.0 / (1.0 + i)
        combined = 0.65 * orig + 0.35 * family_hint_score
        scored_claims.append((combined, family_hint_score, c))

    scored_chunks = []
    for i, ch in enumerate(dedup_chunks):
        txt = str(getattr(ch, "text", "") or "")
        slots = _evidence_slots_text(txt)
        slot_bonus = 0.0
        if any(s in slots for s in {"condition", "mechanism", "outcome"}):
            slot_bonus += 0.12
        if any(s in slots for s in {"management", "prevention", "treatment"}):
            slot_bonus += 0.08
        prov_bonus = 0.05 if str(getattr(ch, "chunk_id", "") or "").strip() else 0.0
        family_hint_score = 0.30 * qa + 0.20 * co + 0.30 * te + 0.10 * rel + slot_bonus + prov_bonus
        orig = 1.0 / (1.0 + i)
        combined = 0.65 * orig + 0.35 * family_hint_score
        scored_chunks.append((combined, family_hint_score, ch))

    scored_paths = []
    for i, p in enumerate(evidence.supporting_paths or []):
        bound = bool(p.text_support_ids)
        if bound:
            trace["bound_graph_hint_count"] += 1
        else:
            trace["unbound_graph_hint_count"] += 1
        orig = 1.0 / (1.0 + i)
        graph_hint_score = (0.25 * qa + 0.20 * co + 0.25 * te + 0.20 * rel + 0.10 * ent)
        if not bound:
            graph_hint_score -= 0.25
        combined = 0.70 * orig + 0.30 * graph_hint_score
        scored_paths.append((combined, graph_hint_score, p))

    scored_claims.sort(key=lambda x: x[0], reverse=True)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    scored_paths.sort(key=lambda x: x[0], reverse=True)

    evidence.supporting_claims = [c for _, _, c in scored_claims]
    evidence.supporting_chunks = [c for _, _, c in scored_chunks]
    # Unbound graph hints stay as hints (kept but last)
    evidence.supporting_paths = [p for _, _, p in scored_paths]

    after_ids = []
    for c in evidence.supporting_claims or []:
        after_ids.append(str(c.get("claim_id") or c.get("relation_id") or c.get("head") or "claim"))
    for ch in evidence.supporting_chunks or []:
        after_ids.append(str(getattr(ch, "chunk_id", None) or "chunk"))
    trace.update(
        {
            "family_hint_rerank_applied": True,
            "evidence_count_after_rerank": len(after_ids),
            "top_evidence_ids_after": after_ids[:10],
            "family_hint_score_distribution": [round(x[1], 4) for x in (scored_claims[:5] + scored_chunks[:5])],
            "exact_duplicate_dropped_count": dropped,
        }
    )
    if not sel:
        trace["rerank_warning"] = "no_selected_family_hint_weak"
    return evidence, trace


def _apply_family_hint_boost_mode(
    evidence,
    family_decision: ClaimFamilyDecision | None,
    mode: str,
    graph_reliance_mode: str,
):
    trace = {
        "fully_auto_family_hint_boost_mode": mode,
        "family_hint_boost_applied": False,
        "evidence_order_changed": False,
        "evidence_annotation_count": 0,
        "high_confidence_annotation_count": 0,
        "medium_confidence_annotation_count": 0,
        "low_confidence_annotation_count": 0,
        "family_hint_alignment_score_summary": {"min": 0.0, "max": 0.0, "mean": 0.0},
        "annotation_warning": "",
    }
    if mode != "evidence_annotation_v1":
        return evidence, trace
    if graph_reliance_mode != "graph_evidence_allowed":
        trace["annotation_warning"] = "mode_limited_to_graph_evidence_allowed"
        return evidence, trace
    if not family_decision:
        trace["annotation_warning"] = "no_family_decision"
        return evidence, trace

    sel = str(family_decision.selected_family or "")
    score_row = (family_decision.family_scores or {}).get(sel, {}) if sel else {}
    qa = float(score_row.get("query_alignment_score", score_row.get("question_focus_alignment", 0.0)) or 0.0)
    co = float(score_row.get("claim_overlap_score", 0.0) or 0.0)
    te = float(score_row.get("text_evidence_support_score", score_row.get("semantic_text_support", 0.0)) or 0.0)
    rel = float(score_row.get("graph_relation_support", score_row.get("graph_support_score", 0.0)) or 0.0)
    ent = float(score_row.get("entity_support_score", 0.0) or 0.0)
    prov = float(score_row.get("cross_source_support_score", score_row.get("cross_source_support", 0.0)) or 0.0)
    slot_keys = ("condition", "mechanism", "outcome", "management", "prevention", "treatment")

    scores: list[float] = []
    hi = 0
    med = 0
    low = 0
    ann = 0
    for c in evidence.supporting_claims or []:
        txt = " ".join(str(c.get(k) or "") for k in ("claim_text", "supporting_quote", "head", "relation", "tail"))
        slots = _evidence_slots_text(txt)
        slot_overlap = sum(1 for s in slot_keys if s in slots)
        relation_overlap = 1.0 if str(c.get("relation_norm") or c.get("relation") or "").strip() else 0.0
        entity_overlap = 1.0 if str(c.get("head") or "").strip() or str(c.get("tail") or "").strip() else 0.0
        prov_avail = 1.0 if str(c.get("source_chunk_id") or c.get("chunk_id") or "").strip() else 0.0
        align = (
            0.26 * qa
            + 0.22 * co
            + 0.22 * te
            + 0.10 * rel
            + 0.10 * ent
            + 0.04 * prov
            + 0.03 * min(slot_overlap, 3) / 3.0
            + 0.02 * relation_overlap
            + 0.01 * entity_overlap
        )
        if prov_avail > 0:
            align += 0.02
        label = "high" if align >= 0.66 else "medium" if align >= 0.45 else "low"
        if label == "high":
            hi += 1
        elif label == "medium":
            med += 1
        else:
            low += 1
        c["family_hint_alignment_score"] = round(align, 4)
        c["family_slot_matches"] = [s for s in slot_keys if s in slots]
        c["family_hint_confidence_label"] = label
        scores.append(align)
        ann += 1

    for ch in evidence.supporting_chunks or []:
        txt = str(getattr(ch, "text", "") or "")
        slots = _evidence_slots_text(txt)
        slot_overlap = sum(1 for s in slot_keys if s in slots)
        prov_avail = 1.0 if str(getattr(ch, "chunk_id", "") or "").strip() else 0.0
        align = (
            0.28 * qa
            + 0.20 * co
            + 0.24 * te
            + 0.10 * rel
            + 0.10 * ent
            + 0.04 * prov
            + 0.04 * min(slot_overlap, 3) / 3.0
        )
        if prov_avail > 0:
            align += 0.02
        label = "high" if align >= 0.66 else "medium" if align >= 0.45 else "low"
        if label == "high":
            hi += 1
        elif label == "medium":
            med += 1
        else:
            low += 1
        setattr(ch, "family_hint_alignment_score", round(align, 4))
        setattr(ch, "family_slot_matches", [s for s in slot_keys if s in slots])
        setattr(ch, "family_hint_confidence_label", label)
        scores.append(align)
        ann += 1

    for p in evidence.supporting_paths or []:
        if not p.text_support_ids:
            continue
        align = 0.24 * qa + 0.20 * co + 0.22 * te + 0.18 * rel + 0.10 * ent + 0.06 * prov
        label = "high" if align >= 0.66 else "medium" if align >= 0.45 else "low"
        if label == "high":
            hi += 1
        elif label == "medium":
            med += 1
        else:
            low += 1
        setattr(p, "family_hint_alignment_score", round(align, 4))
        setattr(p, "family_slot_matches", [])
        setattr(p, "family_hint_confidence_label", label)
        scores.append(align)
        ann += 1

    if scores:
        trace["family_hint_alignment_score_summary"] = {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(sum(scores) / len(scores), 4),
        }
    trace["family_hint_boost_applied"] = True
    trace["evidence_annotation_count"] = ann
    trace["high_confidence_annotation_count"] = hi
    trace["medium_confidence_annotation_count"] = med
    trace["low_confidence_annotation_count"] = low
    if not sel:
        trace["annotation_warning"] = "no_selected_family_annotation_is_weak"
    return evidence, trace


def _should_apply_claim_family_arbitration(question: str, analysis) -> bool:
    ql = (question or "").strip().lower()
    indirect_markers = [
        "despite", "yet", "underlying", "which management limitation", "which reproductive limitation",
        "which nutritional limitation", "which health-related limitation", "which physiological limitation",
        "which factor should be prioritized", "which management factor should be prioritized",
        "which limitation should be evaluated", "which limitation should be considered",
        "which constraint should be considered", "which priority", "remain inconsistent", "remains variable",
    ]
    direct_family_markers = [
        "urolithiasis",
        "white muscle disease",
        "rickets",
        "quarantin",
        "newly purchased goats",
        "castrated male goats",
        "calcium-to-phosphorus",
    ]
    if any(marker in ql for marker in indirect_markers):
        return True
    if any(marker in ql for marker in direct_family_markers):
        return True
    if ql.startswith("why does ") or ql.startswith("why do "):
        if any(
            marker in ql
            for marker in [
                "and what",
                "and how",
                "prevent",
                "reduce the risk",
                "during quarantine",
                "dietary adjustments",
                "sunlight",
                "calcium-to-phosphorus",
                "castrated male goats",
                "newly purchased goats",
                "white muscle disease",
                "urolithiasis",
                "rickets",
            ]
        ):
            return True
        return False
    return analysis.query_type in {"comparison", "multi_hop", "evidence_demanding"} and analysis.answer_slot in {
        "economic", "management", "reproduction", "nutrition", "mechanism", "cause", "connection"
    }


def _run_targeted_arbitration_qe(
    question: str,
    state: AgentState,
    analysis,
    tools: Dict[str, object],
    config: RunConfig,
    targeted_queries: list[dict],
    round_idx: int,
) -> int:
    executed = 0
    for idx, item in enumerate(targeted_queries[:3], start=1):
        tool_name = str(item.get("tool_name") or "").strip() or "TextSearch"
        query = str(item.get("query") or "").strip()
        rationale = str(item.get("rationale") or f"Claim-family arbitration targeted QE round {round_idx}.").strip()
        if not query:
            continue
        subquery_id = f"SQA{round_idx}_{idx}"
        sq = SubQuery(subquery_id=subquery_id, text=query, grounded_text=query, source="ARBITRATION_QE")
        state.subqueries = _merge_subqueries(state.subqueries, [sq])
        step_input = {"query": query, "limit": config.max_chunks, "subquery_id": subquery_id}
        if tool_name == "ClaimSearch":
            step_input["context_terms"] = list((analysis.constraints or {}).get("canonical_terms", []))[:6]
        if tool_name == "HybridSearch":
            step_input["max_hops"] = min(config.max_hops, 2)
        plan = RetrievalPlan([ToolCallSpec(tool_name=tool_name, input=step_input, rationale=rationale)])
        before = len(state.tool_history)
        execute_plan(plan, tools, state, config)
        new_results = state.tool_history[before:]
        channel_ev = refine_context(subquery_id, query, new_results, max_items=config.max_chunks, analysis=analysis)
        state.channel_evidence.append(channel_ev)
        state.tool_trace.append(
            {
                "iteration": state.iterations,
                "action": "arbitration_targeted_qe",
                "round": round_idx,
                "subquery_id": subquery_id,
                "tool_name": tool_name,
                "query": query,
                "semantic_evidence_count": len(channel_ev.semantic_evidence),
                "relational_evidence_count": len(channel_ev.relational_evidence),
            }
        )
        executed += 1
        if state.tool_calls >= config.max_tool_calls:
            break
    return executed


def _apply_claim_family_arbitration(
    question: str,
    analysis,
    state: AgentState,
    evidence,
    report,
    config: RunConfig,
    runtime_skill: dict,
    tools: Dict[str, object],
):
    if not config.enable_claim_family_arbitration:
        return evidence, report, None
    if not _should_apply_claim_family_arbitration(question, analysis):
        return evidence, report, None

    decision = arbitrate_claim_family(
        question,
        analysis,
        evidence,
        state.channel_evidence,
        state.logic_draft,
        report,
        state.reflections,
        runtime_skill,
        arbitration_rounds=0,
        accept_threshold=config.arbitration_accept_threshold,
        refine_threshold=config.arbitration_refine_threshold,
        margin_threshold=config.arbitration_margin_threshold,
    )
    initial_family = decision.selected_family
    state.tool_trace.append(
        {
            "iteration": state.iterations,
            "action": "claim_family_arbitration",
            "round": 0,
            "selected_family": decision.selected_family,
            "top_margin": decision.top_margin,
            "confidence": decision.confidence,
            "conflict_detected": decision.conflict_detected,
            "recommended_targeted_queries": len(decision.recommended_targeted_queries),
        }
    )

    if config.enable_arbitration_targeted_qe:
        rounds = 0
        while (
            rounds < config.max_arbitration_rounds
            and decision.recommended_targeted_queries
            and (
                decision.top_margin < config.arbitration_margin_threshold
                or decision.confidence != "high"
                or (decision.selected_family and decision.family_scores.get(decision.selected_family, {}).get("answer_slot_compatibility", 1.0) < 0.6)
            )
            and state.tool_calls < config.max_tool_calls
        ):
            rounds += 1
            prev_score = evidence.evidence_score
            prev_family = decision.selected_family
            executed = _run_targeted_arbitration_qe(question, state, analysis, tools, config, decision.recommended_targeted_queries, rounds)
            if executed <= 0:
                break
            evidence = build_evidence_package(state.tool_history, analysis)
            evidence = _apply_graph_reliance_mode(
                evidence, state, str((analysis.constraints or {}).get("graph_reliance_mode", "full_current"))
            )
            state.logic_draft = draft_logic(question, state.channel_evidence)
            report = verify_graphsearch_evidence(question, analysis, state.channel_evidence, state.logic_draft, state.iterations, config)
            updated = arbitrate_claim_family(
                question,
                analysis,
                evidence,
                state.channel_evidence,
                state.logic_draft,
                report,
                state.reflections,
                runtime_skill,
                arbitration_rounds=rounds,
                accept_threshold=config.arbitration_accept_threshold,
                refine_threshold=config.arbitration_refine_threshold,
                margin_threshold=config.arbitration_margin_threshold,
            )
            updated.initial_selected_family = initial_family
            updated.family_changed = bool(prev_family and updated.selected_family and updated.selected_family != prev_family)
            decision = updated
            state.tool_trace.append(
                {
                    "iteration": state.iterations,
                    "action": "claim_family_arbitration",
                    "round": rounds,
                    "selected_family": decision.selected_family,
                    "previous_family": prev_family,
                    "family_changed": decision.family_changed,
                    "top_margin": decision.top_margin,
                    "confidence": decision.confidence,
                    "evidence_score_before": prev_score,
                    "evidence_score_after": evidence.evidence_score,
                    "evidence_improved": evidence.evidence_score > prev_score + 1e-9,
                    "conflict_detected": decision.conflict_detected,
                    "recommended_targeted_queries": len(decision.recommended_targeted_queries),
                }
            )
            if decision.confidence == "high" and decision.top_margin >= config.arbitration_margin_threshold:
                break

    if decision.selected_family == "generic_background" and decision.confidence == "low":
        state.tool_trace.append(
            {
                "iteration": state.iterations,
                "action": "claim_family_arbitration_skip",
                "reason": "generic_background_low_confidence",
            }
        )
        return evidence, report, None
    if not decision.selected_family or (
        decision.confidence == "low"
        and report.verdict != "accept"
    ):
        report.verdict = "abstain"
        if "primary_claim_family_uncertain" not in report.missing_information:
            report.missing_information.append("primary_claim_family_uncertain")
    state.family_decision = to_dict(decision)
    return evidence, report, decision


def run_vg_graphrag(
    question: str,
    config: RunConfig,
    graph=None,
    text=None,
    answer_generator=None,
    query_guide=None,
    reflection_generator=None,
    question_id: str | None = None,
) -> FinalAnswer:
    if config.vg_mode != "vg_native_answer":
        raise ValueError("This VG-GraphRAG version only supports vg_native_answer mode.")
    if graph is None or text is None:
        graph, text = build_demo_stores()
    if hasattr(graph, "set_relation_mode"):
        graph.set_relation_mode(config.graph_relation_type, config.allow_fallback_relation_type)

    runtime_skill = load_runtime_skill(config.skill_profile_path) if config.enable_runtime_skill else {}
    state = AgentState(question=question)
    analysis = analyze_query(
        question,
        {
            "ablation_name": config.ablation_name,
            "disable_domain_patterns": config.disable_domain_patterns or config.generic_agentic_only,
            "disable_mechanism_ranking": config.disable_mechanism_ranking or config.generic_agentic_only,
            "disable_scenario_native_synthesis": config.disable_scenario_native_synthesis or config.generic_agentic_only,
            "generic_agentic_only": config.generic_agentic_only,
            "holdout_families": list(config.holdout_families or []),
            "enable_directqa_linking": config.enable_directqa_linking,
            "use_scenario_template_answer": config.use_scenario_template_answer,
            "runtime_skill": runtime_skill,
            "graph_run_id": config.graph_run_id,
            "graph_run_dir": config.graph_run_dir,
            "question_id": question_id,
            "family_registry_mode": config.family_registry_mode,
            "fully_auto_family_gate_mode": config.fully_auto_family_gate_mode,
            "fully_auto_family_runtime_gate_mode": config.fully_auto_family_runtime_gate_mode,
            "fully_auto_family_packaging_mode": config.fully_auto_family_packaging_mode,
            "fully_auto_family_hint_rerank_mode": config.fully_auto_family_hint_rerank_mode,
            "fully_auto_family_hint_boost_mode": config.fully_auto_family_hint_boost_mode,
            "allow_static_family_registry": config.allow_static_family_registry,
            "enable_runtime_candidate_families": config.enable_runtime_candidate_families,
            "use_goat_fallback_family_specs": config.use_goat_fallback_family_specs,
            "evaluation_mode": config.evaluation_mode,
            "evidence_first_scoring": config.evidence_first_scoring,
            "evidence_first_scoring_version": config.evidence_first_scoring_version,
            "graph_as_weak_signal": config.graph_as_weak_signal,
            "graph_reliance_mode": config.graph_reliance_mode,
            "graph_relation_type": config.graph_relation_type,
            "allow_fallback_relation_type": config.allow_fallback_relation_type,
        },
    )
    has_strong_local_hints = bool((analysis.constraints or {}).get("canonical_terms"))
    if query_guide is not None and analysis.query_type in {"multi_hop", "evidence_demanding", "comparison"} and not has_strong_local_hints:
        try:
            guidance = query_guide(question, analysis)
        except Exception:
            guidance = None
        analysis = _merge_query_guidance(analysis, guidance)
    state.analysis = analysis
    tools = _tool_bank(graph, text)
    best_evidence = None
    report = None
    family_decision = None
    subqueries = decompose_relational_queries(decompose_query(question, analysis), analysis)
    state.subqueries = list(subqueries)

    for iteration in range(config.max_iterations):
        state.iterations = iteration
        iteration_channel_evidence = []
        for subquery in subqueries:
            grounded = ground_subquery(subquery, state.channel_evidence)
            plan = create_dual_channel_plan(grounded, analysis, state, config)
            plan = _filter_plan_for_graph_reliance(
                plan, str((analysis.constraints or {}).get("graph_reliance_mode", "full_current"))
            )
            for step in plan.steps:
                step.input["subquery_id"] = grounded.subquery_id
            before = len(state.tool_history)
            execute_plan(plan, tools, state, config)
            new_results = state.tool_history[before:]
            channel_ev = refine_context(grounded.subquery_id, grounded.grounded_text or grounded.text, new_results, max_items=config.max_chunks, analysis=analysis)
            state.channel_evidence.append(channel_ev)
            iteration_channel_evidence.append(channel_ev)
            state.tool_trace.append({
                "iteration": iteration,
                "action": "context_refinement",
                "subquery_id": grounded.subquery_id,
                "semantic_evidence_count": len(channel_ev.semantic_evidence),
                "relational_evidence_count": len(channel_ev.relational_evidence),
                "missing": channel_ev.missing,
            })
            if state.tool_calls >= config.max_tool_calls:
                break
        evidence = build_evidence_package(state.tool_history, analysis)
        evidence = _apply_graph_reliance_mode(
            evidence, state, str((analysis.constraints or {}).get("graph_reliance_mode", "full_current"))
        )
        best_evidence = evidence
        draft = draft_logic(question, state.channel_evidence)
        state.logic_draft = draft
        report = verify_graphsearch_evidence(question, analysis, state.channel_evidence, draft, iteration, config) if config.use_verifier else None
        state.missing_evidence_map = report.missing_evidence_map if report else {}
        verdict = report.verdict if report else "accept"
        state.tool_trace.append({
            "iteration": iteration,
            "action": "verify",
            "verdict": verdict,
            "evidence_score": evidence.evidence_score,
            "failure_modes": report.failure_modes if report else [],
            "missing_evidence_map": report.missing_evidence_map if report else {},
        })
        if verdict == "accept":
            evidence, report, family_decision = _apply_claim_family_arbitration(
                question,
                analysis,
                state,
                evidence,
                report,
                config,
                runtime_skill,
                tools,
            )
            packaged_evidence, packaging_trace = _apply_packaging_mode(
                question,
                analysis,
                evidence,
                family_decision,
                config.fully_auto_family_packaging_mode,
            )
            if family_decision:
                family_decision.candidate_trace.update(packaging_trace)
            reranked_evidence, rerank_trace = _apply_family_hint_rerank_mode(
                packaged_evidence,
                family_decision,
                config.fully_auto_family_hint_rerank_mode,
            )
            if family_decision:
                family_decision.candidate_trace.update(rerank_trace)
            boosted_evidence, boost_trace = _apply_family_hint_boost_mode(
                reranked_evidence,
                family_decision,
                config.fully_auto_family_hint_boost_mode,
                config.graph_reliance_mode,
            )
            if family_decision:
                family_decision.candidate_trace.update(boost_trace)
            ans = synthesize_vg_native_answer(
                question,
                analysis,
                boosted_evidence,
                report,
                state,
                config.diagnostic_only,
                answer_generator=answer_generator if config.use_answer_generator else None,
                use_scenario_template_answer=config.use_scenario_template_answer,
                family_decision=family_decision,
            )  # type: ignore[arg-type]
            ans.family_packaging_trace = packaging_trace
            answer_family_decision = ans.family_decision or {}
            answer_family_decision.setdefault("candidate_trace", {}).update(rerank_trace)
            answer_family_decision.setdefault("candidate_trace", {}).update(boost_trace)
            ans.family_decision = answer_family_decision
            ans.graph_run_id = getattr(graph, "graph_run_id", None)
            return ans
        if verdict == "abstain" or not config.use_refinement_loop:
            break
        reflection = build_self_reflection(question, analysis, state.channel_evidence, draft, report, generator=reflection_generator)
        state.reflections.append(reflection)
        analysis = _merge_reflection(analysis, reflection)
        state.analysis = analysis
        if reflection.missing_evidence_map:
            report.missing_evidence_map = dict(reflection.missing_evidence_map)
            report.missing_information = sorted({m for vals in report.missing_evidence_map.values() for m in vals})
        state.tool_trace.append({
            "iteration": iteration,
            "action": "self_reflection",
            "understanding_question": reflection.understanding_question,
            "analysis_selected_evidence": reflection.analysis_selected_evidence,
            "improved_strategy": reflection.improved_strategy,
            "updated_focus_terms": reflection.updated_focus_terms,
            "preferred_tools": reflection.preferred_tools,
        })
        max_qe = int(skill_flag(runtime_skill, "reflection_policy", "max_qe_subqueries", 3) or 3)
        subqueries = expand_queries(question, state.channel_evidence, report, analysis=analysis, max_new=max_qe)  # type: ignore[arg-type]
        state.subqueries = _merge_subqueries(state.subqueries, subqueries)
        if not subqueries:
            break

    evidence = best_evidence or build_evidence_package(state.tool_history, analysis)
    evidence = _apply_graph_reliance_mode(
        evidence, state, str((analysis.constraints or {}).get("graph_reliance_mode", "full_current"))
    )
    if report is None:
        draft = draft_logic(question, state.channel_evidence)
        state.logic_draft = draft
        report = verify_graphsearch_evidence(question, analysis, state.channel_evidence, draft, config.max_iterations - 1, config)
    if report.verdict != "accept":
        if (
            report.sufficiency_score >= config.accept_threshold
            and not report.failure_modes
            and not report.missing_information
            and evidence.evidence_score >= config.accept_threshold
        ):
            report.verdict = "accept"
        else:
            report.verdict = "abstain"
    evidence, report, family_decision = _apply_claim_family_arbitration(
        question,
        analysis,
        state,
        evidence,
        report,
        config,
        runtime_skill,
        tools,
    )
    packaged_evidence, packaging_trace = _apply_packaging_mode(
        question,
        analysis,
        evidence,
        family_decision,
        config.fully_auto_family_packaging_mode,
    )
    if family_decision:
        family_decision.candidate_trace.update(packaging_trace)
    reranked_evidence, rerank_trace = _apply_family_hint_rerank_mode(
        packaged_evidence,
        family_decision,
        config.fully_auto_family_hint_rerank_mode,
    )
    if family_decision:
        family_decision.candidate_trace.update(rerank_trace)
    boosted_evidence, boost_trace = _apply_family_hint_boost_mode(
        reranked_evidence,
        family_decision,
        config.fully_auto_family_hint_boost_mode,
        config.graph_reliance_mode,
    )
    if family_decision:
        family_decision.candidate_trace.update(boost_trace)
    ans = synthesize_vg_native_answer(
        question,
        analysis,
        boosted_evidence,
        report,
        state,
        config.diagnostic_only,
        answer_generator=answer_generator if config.use_answer_generator else None,
        use_scenario_template_answer=config.use_scenario_template_answer,
        family_decision=family_decision,
    )
    ans.family_packaging_trace = packaging_trace
    answer_family_decision = ans.family_decision or {}
    answer_family_decision.setdefault("candidate_trace", {}).update(rerank_trace)
    answer_family_decision.setdefault("candidate_trace", {}).update(boost_trace)
    ans.family_decision = answer_family_decision
    ans.graph_run_id = getattr(graph, "graph_run_id", None)
    return ans


def _read_qaset(path: Path = Path("data/QASET1.csv")) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _scope_rows(scope: str, qaset: list[dict]) -> list[dict]:
    if scope == "indirect56":
        return [r for r in qaset if 103 <= int(r.get("ID", 0)) <= 158]
    if scope == "full158_diagnostic":
        return qaset
    if scope == "triggered":
        out = []
        for r in qaset:
            q = r.get("Question") or ""
            a = analyze_query(q)
            if a.query_type in {"multi_hop", "evidence_demanding"} or a.answer_slot in {"cause", "treatment", "prevention", "nutrition", "reproduction", "mechanism", "connection"}:
                out.append(r)
        return out
    return qaset[:4]


def run_diagnostic(config: RunConfig) -> dict:
    graph, text, meta = load_graph_run(config.graph_run_id, config.graph_run_dir)
    qaset = _read_qaset()
    rows = _scope_rows(config.case_scope or "triggered", qaset)
    out_dir = Path("data/results")
    cases = []
    traces = []
    packages = []
    verifier_reports = []
    native_answers = []
    for r in rows:
        qid = str(r.get("ID") or r.get("question_id"))
        question = r.get("Question") or r.get("question") or ""
        ans = run_vg_graphrag(question, config, graph=graph, text=text)
        d = to_dict(ans)
        d.update({"question_id": qid, "question": question, "final_answer_overwrite": False})
        cases.append(d)
        traces.append({"question_id": qid, "tool_trace": ans.tool_trace})
        packages.append({"question_id": qid, "supporting_paths": ans.supporting_paths, "supporting_chunks": ans.supporting_chunks})
        verifier_reports.append({"question_id": qid, "verifier_summary": ans.verifier_summary})
        native_answers.append({"question_id": qid, "question": question, "answer_text": ans.answer_text, "confidence": ans.confidence, "abstained": ans.abstained})
    _write_jsonl(out_dir / "vg_graphrag_diagnostic_cases.jsonl", cases)
    _write_jsonl(out_dir / "vg_graphrag_tool_traces.jsonl", traces)
    _write_jsonl(out_dir / "vg_graphrag_evidence_packages.jsonl", packages)
    _write_jsonl(out_dir / "vg_graphrag_verifier_reports.jsonl", verifier_reports)
    _write_jsonl(out_dir / "vg_graphrag_native_answers.jsonl", native_answers)
    summary = _summary(cases, meta, config)
    (out_dir / "vg_graphrag_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "vg_graphrag_report.md").write_text(_report(summary), encoding="utf-8")
    (out_dir / "vg_graphrag_next_step.md").write_text(f"# Next Step\n\nRecommendation: `{summary['recommendation']}`\n", encoding="utf-8")
    return summary


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summary(cases: List[dict], meta: dict, config: RunConfig) -> dict:
    n = len(cases) or 1
    verdicts = Counter((c.get("verifier_summary") or {}).get("verdict", "none") for c in cases)
    actions = Counter(t.get("tool_name") or t.get("action") for c in cases for t in c.get("tool_trace", []))
    scores = [float((c.get("verifier_summary") or {}).get("sufficiency_score", 0) or 0) for c in cases]
    path_tool_hits = 0
    path_tool_calls = 0
    for c in cases:
        for t in c.get("tool_trace", []):
            if t.get("tool_name") == "PathSearch":
                path_tool_calls += 1
                if int(t.get("result_count", 0) or 0) > 0:
                    path_tool_hits += 1
    comp_values: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        # Diagnostic cases only persist final answer state. Recompute component averages from verifier score is unavailable,
        # so keep this explicit empty object rather than implying answer-score semantics.
        pass
    weak = sum(1 for c in cases if "weak" in " ".join(c.get("limitations") or []).lower())
    noise = sum(1 for c in cases if "excessive_noise" in (c.get("limitations") or []))
    native_answer_count = sum(1 for c in cases if not c.get("abstained"))
    abstention_count = sum(1 for c in cases if c.get("abstained"))
    invalid_no_dynamic = sum(1 for c in cases if c.get("invalid_vg_reason") == "invalid_vg_no_dynamic_retrieval")
    v5_usage = sum(1 for c in cases if c.get("used_v5_gate") or c.get("used_v5_outputs") or c.get("v5_usage") != "none")
    recommendation = "proceed_to_limited_judge_native_answers" if native_answer_count and _avg(scores) >= config.accept_threshold else "keep_as_diagnostic_only"
    if invalid_no_dynamic:
        recommendation = "stop_no_dynamic_retrieval"
    elif weak / n > 0.5:
        recommendation = "stop_high_weak_provenance"
    elif _avg(scores) < config.refine_threshold:
        recommendation = "stop_low_evidence_score"
    module_activation = {
        "query_decomposition_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "plan_subquery"
        ),
        "context_refinement_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "context_refinement"
        ),
        "logic_drafting_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "logic_draft"
        ),
        "evidence_verification_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "verify"
        ),
        "query_expansion_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "query_expansion"
        ),
        "self_reflection_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "self_reflection"
        ),
        "arbitration_targeted_qe_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "arbitration_targeted_qe"
        ),
        "claim_family_arbitration_count": sum(
            1 for c in cases for t in c.get("tool_trace", []) if t.get("action") == "claim_family_arbitration"
        ),
    }

    return {
        "vg_mode": "vg_native_answer",
        "processed_case_count": len(cases),
        "graph_run_id": meta.get("graph_run_id"),
        "case_scope": config.case_scope,
        "avg_iterations": _avg([len([t for t in c.get("tool_trace", []) if t.get("action") == "verify"]) for c in cases]),
        "avg_tool_calls": _avg([len([t for t in c.get("tool_trace", []) if t.get("tool_name")]) for c in cases]),
        "action_distribution": dict(actions),
        "verifier_verdict_distribution": dict(verdicts),
        "refinement_triggered_count": sum(1 for c in cases if any((t.get("action") == "verify" and t.get("verdict") == "refine") for t in c.get("tool_trace", []))),
        "native_answer_count": native_answer_count,
        "abstention_count": abstention_count,
        "evidence_score_avg": _avg(scores),
        "evidence_score_components_avg": {k: _avg(v) for k, v in comp_values.items()},
        "path_found_rate": sum(1 for c in cases if c.get("supporting_paths")) / n,
        "path_search_hit_rate": (path_tool_hits / path_tool_calls) if path_tool_calls else 0.0,
        "path_search_call_count": path_tool_calls,
        "text_support_found_rate": sum(1 for c in cases if c.get("supporting_chunks")) / n,
        "weak_provenance_rate": weak / n,
        "excessive_noise_rate": noise / n,
        "no_reference_answer_leakage": True,
        "judge_called": False,
        "independent_dynamic_retrieval_rate": sum(1 for c in cases if c.get("independent_dynamic_retrieval")) / n,
        "invalid_vg_no_dynamic_retrieval_count": invalid_no_dynamic,
        "hop2_context_reuse_count": sum(1 for c in cases if c.get("used_hop2_context_ids")) ,
        "hop2_answer_required_count": sum(1 for c in cases if c.get("used_hop2_answer_as_input")),
        "v5_usage_count": v5_usage,
        "final_answer_overwrite_count": 0,
        "module_activation": module_activation,
        "recommendation": recommendation,
    }


def _report(summary: dict) -> str:
    return "\n".join([
        "# VG-GraphRAG Diagnostic Report",
        "",
        "VG-GraphRAG ran in `vg_native_answer` mode. It did not use GraphRAG-hop2 base answers, hop2 context ids, v5 outputs, or v5 strict gate.",
        "",
        f"- graph_run_id: `{summary.get('graph_run_id')}`",
        f"- processed_case_count: {summary.get('processed_case_count')}",
        f"- independent_dynamic_retrieval_rate: {summary.get('independent_dynamic_retrieval_rate'):.3f}",
        f"- invalid_vg_no_dynamic_retrieval_count: {summary.get('invalid_vg_no_dynamic_retrieval_count')}",
        f"- hop2_answer_required_count: {summary.get('hop2_answer_required_count')}",
        f"- hop2_context_reuse_count: {summary.get('hop2_context_reuse_count')}",
        f"- v5_usage_count: {summary.get('v5_usage_count')}",
        f"- avg_tool_calls: {summary.get('avg_tool_calls'):.3f}",
        f"- verifier verdicts: {summary.get('verifier_verdict_distribution')}",
        f"- native_answer_count: {summary.get('native_answer_count')}",
        f"- abstention_count: {summary.get('abstention_count')}",
        f"- judge_called: {summary.get('judge_called')}",
        f"- final_answer_overwrite_count: {summary.get('final_answer_overwrite_count')}",
        f"- recommendation: `{summary.get('recommendation')}`",
    ]) + "\n"
