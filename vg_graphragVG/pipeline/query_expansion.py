"""Generic query expansion for the VG refinement loop."""

from __future__ import annotations

from vg_graphrag.models import ChannelEvidence, QueryAnalysis, SubQuery, VerifierReport


def expand_queries(
    question: str,
    evidence_items: list[ChannelEvidence],
    report: VerifierReport,
    analysis: QueryAnalysis | None = None,
    max_new: int = 3,
) -> list[SubQuery]:
    """Create bounded follow-up subqueries from verifier gaps."""
    gaps = list(getattr(report, "missing_information", []) or [])
    out: list[SubQuery] = []
    for idx, gap in enumerate(gaps[:max_new]):
        text = f"{question} {gap}".strip()
        out.append(SubQuery(subquery_id=f"qe_{idx}", text=text, channel="dual", source="QE"))
    return out
