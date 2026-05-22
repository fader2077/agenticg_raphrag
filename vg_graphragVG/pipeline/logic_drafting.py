from __future__ import annotations

import re

from vg_graphrag.models import ChannelEvidence, LogicDraft


def _short_answer(ev: ChannelEvidence) -> str:
    text = ev.semantic_summary or ev.relational_summary
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return ""
    return text[:220]


def draft_logic(question: str, evidence_items: list[ChannelEvidence]) -> LogicDraft:
    """LD: build a traceable reasoning draft from refined contexts."""
    steps: list[str] = []
    answers: dict[str, str] = {}
    gaps: dict[str, list[str]] = {}
    for ev in evidence_items:
        answer = _short_answer(ev)
        if answer:
            steps.append(f"{ev.subquery_id}: {answer}")
            answers[ev.subquery_id] = answer
        if ev.missing:
            gaps[ev.subquery_id] = list(ev.missing)
    draft = " ".join(steps)
    if not draft:
        draft = "No sufficiently grounded reasoning draft could be formed."
    return LogicDraft(steps, answers, gaps, draft)
