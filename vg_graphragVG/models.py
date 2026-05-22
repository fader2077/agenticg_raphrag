from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)


@dataclass
class Node:
    node_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    node_type: str = "entity"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    edge_id: str
    source: str
    target: str
    relation: str
    supporting_quote: str = ""
    source_chunk_id: Optional[str] = None
    source_document_id: Optional[str] = None
    confidence: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    source_document_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryEntity:
    text: str
    node_id: Optional[str] = None
    score: float = 0.0
    entity_type: str = "unknown"


@dataclass
class QueryAnalysis:
    query_type: str
    entities: List[QueryEntity]
    constraints: Dict[str, Any]
    expected_hops: int
    needs_text_evidence: bool
    requires_citations: bool
    answer_slot: str


@dataclass
class ToolCallSpec:
    tool_name: str
    input: Dict[str, Any]
    rationale: str
    depends_on: List[str] = field(default_factory=list)
    max_hops: Optional[int] = None
    relation_filters: List[str] = field(default_factory=list)


@dataclass
class ToolResult:
    tool_name: str
    query: Dict[str, Any]
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cost_metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    steps: List[ToolCallSpec]


@dataclass
class SubQuery:
    subquery_id: str
    text: str
    channel: str = "dual"
    grounded_text: str = ""
    relation_triples: List[Dict[str, str]] = field(default_factory=list)
    source: str = "QD"


@dataclass
class ChannelEvidence:
    subquery_id: str
    grounded_query: str
    semantic_evidence: List[Dict[str, Any]] = field(default_factory=list)
    relational_evidence: List[Dict[str, Any]] = field(default_factory=list)
    semantic_summary: str = ""
    relational_summary: str = ""
    missing: List[str] = field(default_factory=list)


@dataclass
class LogicDraft:
    reasoning_steps: List[str] = field(default_factory=list)
    intermediate_answers: Dict[str, str] = field(default_factory=dict)
    evidence_gaps: Dict[str, List[str]] = field(default_factory=dict)
    draft_answer: str = ""


@dataclass
class SelfReflectionReport:
    understanding_question: str
    analysis_selected_evidence: str
    improved_strategy: str
    updated_focus_terms: List[str] = field(default_factory=list)
    avoid_terms: List[str] = field(default_factory=list)
    preferred_tools: List[str] = field(default_factory=list)
    missing_evidence_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class EvidencePath:
    nodes: List[str]
    edges: List[Dict[str, Any]]
    text_support_ids: List[str] = field(default_factory=list)


@dataclass
class EvidencePackage:
    claim_candidates: List[Dict[str, Any]]
    supporting_claims: List[Dict[str, Any]]
    supporting_paths: List[EvidencePath]
    supporting_chunks: List[TextChunk]
    subgraph_nodes: List[Node]
    subgraph_edges: List[Edge]
    coverage_flags: Dict[str, bool]
    evidence_score: float
    score_components: Dict[str, float]
    noise_flags: Dict[str, bool]
    provenance_summary: Dict[str, Any]


@dataclass
class VerifierReport:
    verdict: str
    sufficiency_score: float
    failure_modes: List[str]
    missing_information: List[str]
    recommended_actions: List[ToolCallSpec]
    rationale: str
    missing_evidence_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ClaimFamilyDecision:
    candidate_families: List[str]
    selected_family: Optional[str]
    rejected_families: List[str]
    family_scores: Dict[str, Dict[str, float]]
    top_margin: float
    conflict_detected: bool
    missing_disambiguating_evidence: List[str] = field(default_factory=list)
    recommended_targeted_queries: List[Dict[str, Any]] = field(default_factory=list)
    arbitration_rounds: int = 0
    confidence: str = "low"
    rationale: str = ""
    initial_selected_family: Optional[str] = None
    family_changed: bool = False
    family_scope: str = "unknown"
    static_family_count: int = 0
    runtime_candidate_family_count: int = 0
    selected_static_family_count: int = 0
    selected_runtime_family_count: int = 0
    selected_family_source: str = "unknown"
    candidate_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    question: str
    analysis: Optional[QueryAnalysis] = None
    tool_history: List[ToolResult] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    subqueries: List[SubQuery] = field(default_factory=list)
    channel_evidence: List[ChannelEvidence] = field(default_factory=list)
    logic_draft: Optional[LogicDraft] = None
    missing_evidence_map: Dict[str, List[str]] = field(default_factory=dict)
    reflections: List[SelfReflectionReport] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0


@dataclass
class FinalAnswer:
    answer_text: str
    confidence: str
    supporting_paths: List[Dict[str, Any]]
    supporting_chunks: List[Dict[str, Any]]
    limitations: List[str]
    verifier_summary: Dict[str, Any]
    tool_trace: List[Dict[str, Any]]
    graph_run_id: Optional[str] = None
    diagnostic_only: bool = True
    vg_mode: str = "vg_native_answer"
    independent_dynamic_retrieval: bool = False
    dynamic_tool_call_count: int = 0
    used_hop2_context_ids: bool = False
    used_hop2_answer_as_input: bool = False
    used_v5_gate: bool = False
    used_v5_outputs: bool = False
    hop2_usage: str = "none"
    v5_usage: str = "none"
    invalid_vg_reason: Optional[str] = None
    abstained: bool = False
    subqueries: List[Dict[str, Any]] = field(default_factory=list)
    channel_evidence: List[Dict[str, Any]] = field(default_factory=list)
    logic_draft: Dict[str, Any] = field(default_factory=dict)
    supporting_claims: List[Dict[str, Any]] = field(default_factory=list)
    reflections: List[Dict[str, Any]] = field(default_factory=list)
    family_decision: Dict[str, Any] = field(default_factory=dict)
    family_packaging_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    vg_mode: str = "vg_native_answer"
    max_iterations: int = 2
    max_tool_calls: int = 12
    max_hops: int = 3
    max_chunks: int = 5
    accept_threshold: float = 0.62
    refine_threshold: float = 0.35
    use_graph_tools: bool = True
    use_text_tools: bool = True
    use_verifier: bool = True
    use_refinement_loop: bool = True
    fixed_k_hop_only: bool = False
    diagnostic_only: bool = True
    graph_run_id: Optional[str] = None
    case_scope: Optional[str] = None
    graph_run_dir: Optional[str] = None
    ablation_name: str = "VG-full"
    disable_domain_patterns: bool = False
    disable_mechanism_ranking: bool = False
    disable_scenario_native_synthesis: bool = False
    generic_agentic_only: bool = False
    holdout_families: List[str] = field(default_factory=list)
    enable_directqa_linking: bool = False
    use_scenario_template_answer: bool = False
    enable_runtime_skill: bool = True
    skill_profile_path: Optional[str] = None
    use_answer_generator: bool = True
    enable_claim_family_arbitration: bool = False
    enable_arbitration_targeted_qe: bool = False
    max_arbitration_rounds: int = 3
    arbitration_accept_threshold: float = 0.70
    arbitration_refine_threshold: float = 0.60
    arbitration_margin_threshold: float = 0.15
    family_registry_mode: str = "seeded_registry_current"
    fully_auto_family_gate_mode: str = "off"
    fully_auto_family_runtime_gate_mode: str = "off"
    fully_auto_family_packaging_mode: str = "off"
    fully_auto_family_hint_rerank_mode: str = "off"
    fully_auto_family_hint_boost_mode: str = "off"
    allow_static_family_registry: bool = True
    enable_runtime_candidate_families: bool = True
    use_goat_fallback_family_specs: bool = True
    evaluation_mode: bool = True
    evidence_first_scoring: bool = False
    evidence_first_scoring_version: str = "v1"
    graph_as_weak_signal: bool = False
    graph_reliance_mode: str = "full_current"
    graph_relation_type: str = "RELATION"
    allow_fallback_relation_type: bool = False
