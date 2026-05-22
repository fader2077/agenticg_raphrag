from __future__ import annotations

from vg_graphrag.models import EvidencePackage, RetrievalPlan, RunConfig, VerifierReport


def refine_plan(current_plan: RetrievalPlan, evidence: EvidencePackage, report: VerifierReport, config: RunConfig) -> RetrievalPlan:
    steps = []
    for action in report.recommended_actions:
        inp = dict(action.input)
        if "max_hops" in inp:
            inp["max_hops"] = min(int(inp["max_hops"]) + 1, config.max_hops)
        action.input = inp
        steps.append(action)
    if not steps:
        steps = current_plan.steps[-1:]
    return RetrievalPlan(steps=steps)
