from __future__ import annotations

from vg_graphrag.models import EvidencePackage, QueryAnalysis, RunConfig, ToolCallSpec, VerifierReport


def verify_evidence(question: str, analysis: QueryAnalysis, evidence: EvidencePackage, iteration: int, config: RunConfig) -> VerifierReport:
    failures: list[str] = []
    missing: list[str] = []
    actions: list[ToolCallSpec] = []
    flags = evidence.coverage_flags
    if not flags.get("source_entity_found"):
        failures.append("missing_source_entity")
        missing.append("source entity")
        actions.append(ToolCallSpec("EntitySearch", {"query": question, "context_terms": [e.text for e in analysis.entities]}, "Retry entity grounding with context terms."))
    if analysis.query_type in {"multi_hop", "evidence_demanding"} and not flags.get("path_found"):
        failures.append("missing_reasoning_path")
        missing.append("graph reasoning path")
        actions.append(ToolCallSpec("PathSearch", {"source_query": analysis.entities[0].text if analysis.entities else question, "target_query": analysis.entities[-1].text if analysis.entities else question, "max_hops": min(config.max_hops, 3)}, "Try deeper path search."))
    if flags.get("path_found") and not flags.get("text_support_found"):
        failures.append("missing_text_support")
        missing.append("text support for graph evidence")
        actions.append(ToolCallSpec("TextSearch", {"query": question, "limit": config.max_chunks}, "Retrieve text support for graph path."))
    if evidence.noise_flags.get("weak_provenance"):
        failures.append("weak_provenance")
        actions.append(ToolCallSpec("HybridSearch", {"query": question, "limit": config.max_chunks}, "Find provenance-backed chunks linked to graph evidence."))

    hard = {"missing_source_entity", "missing_reasoning_path", "missing_text_support", "weak_provenance"} & set(failures)
    single_hop_soft_accept = (
        analysis.query_type == "single_hop"
        and flags.get("source_entity_found")
        and flags.get("text_support_found")
        and evidence.evidence_score >= max(0.45, config.refine_threshold)
        and not {"missing_source_entity", "missing_text_support"} & set(failures)
    )
    if (evidence.evidence_score >= config.accept_threshold and not hard) or single_hop_soft_accept:
        verdict = "accept"
    elif evidence.evidence_score >= config.refine_threshold and iteration + 1 < config.max_iterations:
        verdict = "refine"
    else:
        verdict = "abstain" if iteration + 1 >= config.max_iterations else "refine"
    return VerifierReport(verdict, evidence.evidence_score, failures, missing, actions, f"score={evidence.evidence_score:.3f}; failures={','.join(failures) or 'none'}")
