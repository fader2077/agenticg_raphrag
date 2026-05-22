from __future__ import annotations

import re

from vg_graphrag.models import QueryAnalysis, SubQuery


def _clean(text: str) -> str:
    return " ".join((text or "").replace("?", " ").split()).strip()


def _split_clauses(question: str) -> list[str]:
    q = _clean(question)
    parts = re.split(r"\b(?:and|then|after|before|because|through|via|while|despite|yet|but|although|whereas)\b|[,;:]", q, flags=re.I)
    out = []
    for p in parts:
        s = _clean(p)
        if len(s.split()) >= 3 and s.lower() not in {x.lower() for x in out}:
            out.append(s)
    return out


def decompose_query(question: str, analysis: QueryAnalysis, max_subqueries: int = 4) -> list[SubQuery]:
    """QD: deterministic subquery decomposition for tests and offline runs.

    This is intentionally lightweight: it creates retrieval-oriented subqueries
    rather than reasoning prose, and does not use reference answers or judges.
    """
    clauses = _split_clauses(question)
    subqueries: list[SubQuery] = []
    focus = analysis.constraints.get("diagnostic_focus", []) if analysis.constraints else []
    focus_text = "; ".join(str(x) for x in focus[:2])
    hint_terms = analysis.constraints.get("canonical_terms", []) if analysis.constraints else []
    hint_text = ", ".join(str(x) for x in hint_terms[:5])
    if analysis.query_type in {"multi_hop", "evidence_demanding", "comparison"} and clauses:
        for idx, clause in enumerate(clauses[:max_subqueries], 1):
            subqueries.append(SubQuery(f"SQ{idx}", clause, channel="dual", grounded_text=clause))
    else:
        subqueries.append(SubQuery("SQ1", question, channel="dual", grounded_text=question))

    if analysis.query_type == "evidence_demanding" and hint_terms and len(subqueries) < max_subqueries:
        diag_query = (
            f"Retrieve claim-level, graph, and text evidence for these candidate concepts: {hint_text}. "
            f"Scenario: {question} Focus on: {focus_text or 'the main underlying mechanism'}."
        )
        subqueries.insert(0, SubQuery("SQ0", diag_query, channel="dual", grounded_text=diag_query))

    ql = question.lower()
    if (
        hint_terms
        and len(subqueries) < max_subqueries
        and any(x in ql for x in ["prioritized", "prioritize", "limitation", "constraint", "management review", "should be evaluated"])
    ):
        diag_query = (
            f"Retrieve the most specific evidence for the underlying {analysis.answer_slot} issue using these candidate concepts: {hint_text}. "
            f"Scenario: {question} Focus on: {focus_text or 'the central mechanism'}."
        )
        subqueries.insert(0, SubQuery("SQD", diag_query, channel="dual", grounded_text=diag_query))

    # Add an explicit evidence/support subquery when the original question is broad or indirect.
    if analysis.query_type in {"multi_hop", "evidence_demanding"} and len(subqueries) < max_subqueries:
        ents = [e.text for e in analysis.entities[:3]]
        anchor = " and ".join(ents) if ents else question
        text = f"What text and graph evidence supports the relationship or answer for {anchor}?"
        if hint_text:
            text += f" Prioritize concepts such as {hint_text}."
        subqueries.append(SubQuery(f"SQ{len(subqueries) + 1}", text, channel="dual", grounded_text=text))
    return subqueries[:max_subqueries]


def decompose_relational_queries(subqueries: list[SubQuery], analysis: QueryAnalysis) -> list[SubQuery]:
    """Create simple subject-relation-object hints for the relational channel."""
    ents = [e.text for e in analysis.entities]
    for sq in subqueries:
        if len(ents) >= 2:
            sq.relation_triples.append({"subject": ents[0], "predicate": analysis.answer_slot, "object": ents[-1]})
        elif ents:
            sq.relation_triples.append({"subject": ents[0], "predicate": analysis.answer_slot, "object": ""})
    return subqueries
