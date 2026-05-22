from __future__ import annotations

import re
from typing import Callable

from vg_graphrag.models import ChannelEvidence, LogicDraft, QueryAnalysis, SelfReflectionReport, VerifierReport


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 2]


def _focus_terms(analysis: QueryAnalysis) -> list[str]:
    out: list[str] = []
    if analysis.constraints:
        out.extend(str(x) for x in analysis.constraints.get("canonical_terms", []))
        out.extend(str(x) for x in analysis.constraints.get("alias_terms", []))
    seen = set()
    uniq = []
    for item in out:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq[:8]


def build_self_reflection(
    question: str,
    analysis: QueryAnalysis,
    evidence_items: list[ChannelEvidence],
    draft: LogicDraft,
    report: VerifierReport,
    generator: Callable[[str, QueryAnalysis, list[ChannelEvidence], LogicDraft, VerifierReport], dict] | None = None,
) -> SelfReflectionReport:
    if generator is not None:
        try:
            data = generator(question, analysis, evidence_items, draft, report) or {}
            return SelfReflectionReport(
                understanding_question=str(data.get("understanding_question") or "").strip(),
                analysis_selected_evidence=str(data.get("analysis_selected_evidence") or "").strip(),
                improved_strategy=str(data.get("improved_strategy") or "").strip(),
                updated_focus_terms=[str(x) for x in data.get("updated_focus_terms", []) if str(x).strip()],
                avoid_terms=[str(x) for x in data.get("avoid_terms", []) if str(x).strip()],
                preferred_tools=[str(x) for x in data.get("preferred_tools", []) if str(x).strip()],
                missing_evidence_map={str(k): [str(x) for x in v] for k, v in (data.get("missing_evidence_map") or {}).items()},
            )
        except Exception:
            pass

    focus_terms = _focus_terms(analysis)
    semantic_only = [ev.subquery_id for ev in evidence_items if ev.semantic_evidence and not ev.relational_evidence]
    no_focus = []
    for ev in evidence_items:
        joined = " ".join([ev.semantic_summary, ev.relational_summary]).lower()
        if focus_terms and not any(term.replace("_", " ").lower() in joined for term in focus_terms):
            no_focus.append(ev.subquery_id)
    understanding = (
        f"The question is primarily asking for the main {analysis.answer_slot.replace('_', ' ')}"
        f" and not generic background. Expected query type: {analysis.query_type}."
    )
    selected = (
        "The current retrieval over-relied on generic text evidence."
        if semantic_only or no_focus
        else "The current retrieval already contains both semantic and relational evidence, but the mechanism link is still incomplete."
    )
    if semantic_only:
        selected += f" Semantic-only subqueries: {', '.join(semantic_only[:4])}."
    if no_focus:
        selected += f" Focus-misaligned evidence appeared in: {', '.join(no_focus[:4])}."
    missing_modes = sorted({m for vals in report.missing_evidence_map.values() for m in vals})
    preferred_tools = ["ClaimSearch", "PathSearch", "HybridSearch"]
    updated_terms = focus_terms[:6]
    avoid_terms = ["generic_management", "generic_breed_background"] if semantic_only or no_focus else []
    if "mechanism_claim_evidence" in missing_modes:
        preferred_tools = ["ClaimSearch", "HybridSearch", "PathSearch", "TextSearch"]
    elif "relational_graph_evidence" in missing_modes or "missing_reasoning_path" in report.failure_modes:
        preferred_tools = ["PathSearch", "GraphNeighbor", "HybridSearch", "ClaimSearch"]
    elif "semantic_text_evidence" in missing_modes:
        preferred_tools = ["TextSearch", "HybridSearch", "ClaimSearch"]
    if "focus_aligned_evidence" in missing_modes:
        updated_terms = updated_terms + ["underlying mechanism", analysis.answer_slot]
        avoid_terms = avoid_terms + ["generic direct answer", "broad background"]
    reflection_missing_map = dict(report.missing_evidence_map or {})
    if not reflection_missing_map:
        for sid in semantic_only[:4]:
            reflection_missing_map.setdefault(sid, [])
            for gap in ["relational_graph_evidence", "mechanism_claim_evidence"]:
                if gap not in reflection_missing_map[sid]:
                    reflection_missing_map[sid].append(gap)
        for sid in no_focus[:4]:
            reflection_missing_map.setdefault(sid, [])
            for gap in ["focus_aligned_evidence", "semantic_text_evidence"]:
                if gap not in reflection_missing_map[sid]:
                    reflection_missing_map[sid].append(gap)
    improved = (
        "Prioritize mechanism-bearing claim evidence first, then require a bounded graph path and explicit text support for that mechanism. "
        "Avoid generic breed, housing, or management background unless it explicitly names the target limitation."
    )
    return SelfReflectionReport(
        understanding_question=understanding,
        analysis_selected_evidence=selected,
        improved_strategy=improved,
        updated_focus_terms=updated_terms,
        avoid_terms=avoid_terms,
        preferred_tools=preferred_tools,
        missing_evidence_map=reflection_missing_map,
    )
