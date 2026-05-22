from __future__ import annotations

import re
from typing import List

from vg_graphrag.domain import build_domain_hints
from vg_graphrag.models import QueryAnalysis, QueryEntity


SLOTS = {
    "cause": ["cause", "causes", "why", "mechanism"],
    "treatment": ["treat", "treatment", "therapy", "cure"],
    "prevention": ["prevent", "prevention", "avoid"],
    "nutrition": ["feed", "nutrition", "nutrient", "deficiency", "colostrum"],
    "reproduction": ["pregnancy", "reproduction", "breeding", "estrus"],
    "breed_trait": ["breed", "trait"],
    "management": ["manage", "management", "housing", "care"],
    "economic": ["cost", "profit", "economic", "income"],
    "mechanism": ["mechanism", "pathway", "through"],
    "connection": ["connected", "connection", "indirectly", "path"],
}


def _answer_slot(question: str) -> str:
    q = question.lower()
    for slot, terms in SLOTS.items():
        if any(t in q for t in terms):
            return slot
    return "other"


def _entities(question: str) -> List[QueryEntity]:
    ents = []
    seen = set()
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9_]*(?:[A-Z][A-Za-z0-9_]*)?\b", question):
        txt = m.group(0)
        if txt.lower() in {"what", "which", "how", "does", "following", "after", "during", "despite", "with"}:
            continue
        if txt.lower() not in seen:
            seen.add(txt.lower())
            ents.append(QueryEntity(text=txt))
    # Lowercase domain entity fallback for questions without capitalized entities.
    token_fallback = [
        "goat",
        "kid",
        "newborn",
        "doe",
        "buck",
        "colostrum",
        "mastitis",
        "diarrhea",
        "pneumonia",
        "parasite",
        "deworm",
        "feed",
        "nutrition",
        "vitamin",
        "calcium",
        "phosphorus",
        "breeding",
        "estrus",
        "housing",
        "ventilation",
        "trial",
        "company",
    ]
    ql = question.lower()
    for tok in token_fallback:
        if tok in ql and tok not in seen:
            seen.add(tok)
            ents.append(QueryEntity(text=tok, entity_type="domain_term"))
    return ents


def analyze_query(question: str, graph_schema_metadata: dict | None = None) -> QueryAnalysis:
    q = question.lower()
    metadata = graph_schema_metadata or {}
    hints = build_domain_hints(
        question,
        exclude_patterns=list(metadata.get("holdout_families", []) or []),
        disable=bool(metadata.get("disable_domain_patterns", False)),
        enable_directqa_linking=bool(metadata.get("enable_directqa_linking", False)),
    )
    if any(x in q for x in ["connected to", "through", "indirectly", "path", "mechanism"]):
        query_type = "multi_hop"
        hops = 2
    elif any(x in q for x in ["despite", "yet", "but", "which physiological limitation", "which management", "which dietary", "which nutritional", "should be prioritized", "most likely"]):
        query_type = "evidence_demanding"
        hops = 2
    elif any(x in q for x in ["what evidence supports", "evidence", "supporting evidence"]):
        query_type = "evidence_demanding"
        hops = 2
    elif any(x in q for x in ["compare", "difference", "versus", " vs "]):
        query_type = "comparison"
        hops = 1
    elif any(x in q for x in ["before", "after", "during", "when"]):
        query_type = "temporal"
        hops = 1
    elif any(x in q for x in ["how many", "average", "total"]):
        query_type = "aggregation"
        hops = 1
    else:
        query_type = "single_hop"
        hops = 1
    entities = _entities(question)
    seen = {e.text.lower() for e in entities}
    for term in hints.get("alias_terms", []):
        t = str(term)
        if t.lower() not in seen:
            seen.add(t.lower())
            entities.append(QueryEntity(text=t, entity_type="canonical_hint"))
    return QueryAnalysis(
        query_type=query_type,
        entities=entities,
        constraints={
            "domain_hints": hints,
            "canonical_terms": hints.get("canonical_terms", []),
            "alias_terms": hints.get("alias_terms", []),
            "target_entities": hints.get("alias_terms", []),
            "directqa_ids": hints.get("directqa_ids", []),
            "diagnostic_focus": hints.get("diagnostic_focus", []),
            "relation_terms": hints.get("relation_terms", []),
            "disable_mechanism_ranking": bool(metadata.get("disable_mechanism_ranking", False)),
            "disable_scenario_native_synthesis": bool(metadata.get("disable_scenario_native_synthesis", False)),
            "generic_agentic_only": bool(metadata.get("generic_agentic_only", False)),
            "enable_directqa_linking": bool(metadata.get("enable_directqa_linking", False)),
            "use_scenario_template_answer": bool(metadata.get("use_scenario_template_answer", False)),
            "runtime_skill": dict(metadata.get("runtime_skill") or {}),
            "ablation_name": str(metadata.get("ablation_name", "VG-full")),
            "graph_run_id": metadata.get("graph_run_id"),
            "graph_run_dir": metadata.get("graph_run_dir"),
            "question_id": metadata.get("question_id"),
            "family_registry_mode": str(metadata.get("family_registry_mode", "seeded_registry_current")),
            "allow_static_family_registry": bool(metadata.get("allow_static_family_registry", True)),
            "enable_runtime_candidate_families": bool(metadata.get("enable_runtime_candidate_families", True)),
            "use_goat_fallback_family_specs": bool(metadata.get("use_goat_fallback_family_specs", True)),
            "evaluation_mode": bool(metadata.get("evaluation_mode", True)),
            "evidence_first_scoring": bool(metadata.get("evidence_first_scoring", False)),
            "evidence_first_scoring_version": str(metadata.get("evidence_first_scoring_version", "v1")),
            "graph_as_weak_signal": bool(metadata.get("graph_as_weak_signal", False)),
            "graph_reliance_mode": str(metadata.get("graph_reliance_mode", "full_current")),
        },
        expected_hops=hops,
        needs_text_evidence=True,
        requires_citations=("evidence" in q or "support" in q),
        answer_slot=_answer_slot(question),
    )
