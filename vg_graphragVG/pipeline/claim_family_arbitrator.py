"""Minimal claim-family arbitration compatibility for VG GraphRAG."""

from __future__ import annotations

from typing import Any

from vg_graphrag.models import ClaimFamilyDecision, EvidencePackage, QueryAnalysis, VerifierReport


def filter_evidence_for_family(evidence: EvidencePackage, selected_family: str | None = None) -> EvidencePackage:
    """Return evidence unchanged when no domain family registry is configured."""
    return evidence


def arbitrate_claim_family(
    question: str,
    analysis: QueryAnalysis,
    evidence: EvidencePackage,
    report: VerifierReport,
    runtime_skill: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ClaimFamilyDecision:
    """Choose a generic evidence family without answer leakage."""
    family = str(getattr(analysis, "answer_slot", "") or "generic")
    score = float(getattr(evidence, "evidence_score", 0.0) or 0.0)
    return ClaimFamilyDecision(
        candidate_families=[family],
        selected_family=family,
        rejected_families=[],
        family_scores={family: {"query_alignment_score": score, "claim_overlap_score": score, "text_evidence_support_score": score}},
        top_margin=1.0,
        conflict_detected=False,
        confidence="medium" if score >= 0.5 else "low",
        rationale="Generic HotpotQA-compatible arbitration fallback.",
        selected_family_source="runtime_generic",
        runtime_candidate_family_count=1,
        selected_runtime_family_count=1,
        candidate_trace={"fallback": "generic_claim_family_arbitrator"},
    )
