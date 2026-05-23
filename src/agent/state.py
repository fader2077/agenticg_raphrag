"""Agent state dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable state for one bounded AgenticGraphRAG run."""

    qid: str
    question: str
    pipeline_version: str = "hotpotqa_agentic_graphrag_v1"
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    repair_trace: list[dict[str, Any]] = field(default_factory=list)
    verifier_trace: list[dict[str, Any]] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieved_entities: list[str] = field(default_factory=list)
    retrieved_paths: list[dict[str, Any]] = field(default_factory=list)
    verifier: dict[str, Any] = field(default_factory=dict)
    repair_rounds: int = 0
    analysis: dict[str, Any] = field(default_factory=dict)
    vg_graphrag_integration: dict[str, Any] = field(
        default_factory=lambda: {
            "vg_graphrag_reference_used": False,
            "vg_graphrag_modules_called": [],
            "vg_graphrag_adapters_used": [],
            "vg_graphrag_abstractions_embedded": [],
            "controller_backend": "bounded_deterministic_v1",
            "integration_status": "not_initialized",
        }
    )
