"""Skill abstractions for the HotpotQA VG-compatible AgenticGraphRAG controller."""

from __future__ import annotations

from typing import Any

from src.agent.evidence_fusion import fuse_evidence
from src.agent.vg_adapter import run_vg_planner_pass
from src.generation.answer_generator import AnswerGenerator


class QuestionAnalysisSkill:
    """Question analysis through reusable VG analyzer/planner scaffolding."""

    def __init__(self, text_retriever: Any, graph_retriever: Any, config: dict[str, Any]):
        self.text_retriever = text_retriever
        self.graph_retriever = graph_retriever
        self.config = config

    def run(self, question: str, state: Any) -> dict[str, Any]:
        agent_cfg = self.config.get("agent", {})
        graph_cfg = self.config.get("graph", {})
        bundle = run_vg_planner_pass(
            question=question,
            question_id=state.qid,
            text_retriever=self.text_retriever,
            graph_retriever=self.graph_retriever,
            skill_profile_path=agent_cfg.get("vg_skill_profile_path"),
            graph_run_id=str(graph_cfg.get("graph_run_id", "")),
            max_tool_calls=int(agent_cfg.get("max_tool_calls", 10)),
            max_hops=int(agent_cfg.get("max_graph_depth", 2)),
            max_chunks=int(self.config.get("retrieval", {}).get("top_k_text", 10)),
        )
        return {
            "query_type": bundle.analysis.query_type,
            "answer_slot": bundle.analysis.answer_slot,
            "expected_hops": bundle.analysis.expected_hops,
            "entities": [e.text for e in bundle.analysis.entities],
            "vg_bundle": bundle,
        }


class TextRetrievalSkill:
    """Semantic retrieval over the text index."""

    def __init__(self, retriever: Any, retrieval_cfg: dict[str, Any]):
        self.retriever = retriever
        self.retrieval_cfg = retrieval_cfg

    def run(self, question: str, state: Any) -> list[dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=int(self.retrieval_cfg.get("top_k_text", 10)))


class VectorRetrievalSkill:
    """Dense retrieval over the vector index."""

    def __init__(self, retriever: Any, retrieval_cfg: dict[str, Any]):
        self.retriever = retriever
        self.retrieval_cfg = retrieval_cfg

    def run(self, question: str, state: Any) -> list[dict[str, Any]]:
        return self.retriever.retrieve(question, top_k=int(self.retrieval_cfg.get("top_k_vector", 10)))


class GraphRetrievalSkill:
    """Explicit graph retrieval with bounded hop expansion."""

    def __init__(self, retriever: Any, retrieval_cfg: dict[str, Any], agent_cfg: dict[str, Any]):
        self.retriever = retriever
        self.retrieval_cfg = retrieval_cfg
        self.agent_cfg = agent_cfg

    def run(self, question: str, state: Any) -> dict[str, Any]:
        expected_hops = 1
        analysis = getattr(state, "analysis", None) or {}
        if isinstance(analysis, dict):
            expected_hops = int(analysis.get("expected_hops", 1))
        depth = min(int(self.agent_cfg.get("max_graph_depth", 2)), max(1, expected_hops))
        return self.retriever.retrieve(
            question,
            depth=depth,
            top_k_paths=int(self.retrieval_cfg.get("top_k_graph_paths", 5)),
            max_nodes_per_hop=int(self.retrieval_cfg.get("max_nodes_per_hop", 10)),
        )


class EvidenceFusionSkill:
    """Fuse text, vector, and graph evidence into one bounded evidence set."""

    def run(
        self,
        text_chunks: list[dict[str, Any]],
        vector_chunks: list[dict[str, Any]],
        graph_chunks: list[dict[str, Any]],
        state: Any,
    ) -> list[dict[str, Any]]:
        return fuse_evidence(text_chunks, vector_chunks, graph_chunks, limit=20)


class GroundedAnsweringSkill:
    """Generate the final answer from bounded evidence."""

    def __init__(self, generator: AnswerGenerator):
        self.generator = generator

    def run(self, question: str, evidence: list[dict[str, Any]], state: Any) -> dict[str, Any]:
        return self.generator.generate(question, evidence, method="AgenticGraphRAG")


class RepairSkill:
    """Run one bounded repair cycle when verification fails."""

    def __init__(self, text_retriever: Any, answering_skill: GroundedAnsweringSkill, retrieval_cfg: dict[str, Any]):
        self.text_retriever = text_retriever
        self.answering_skill = answering_skill
        self.retrieval_cfg = retrieval_cfg

    def run(self, question: str, answer: str, evidence: list[dict[str, Any]], state: Any) -> dict[str, Any]:
        repair_query = f"{question} {answer}".strip()
        repair_chunks = self.text_retriever.retrieve(repair_query, top_k=int(self.retrieval_cfg.get("top_k_text", 10)))
        repaired_evidence = fuse_evidence(evidence, repair_chunks, limit=20)
        generation = self.answering_skill.run(question, repaired_evidence, state)
        return {
            "repair_query": repair_query,
            "repair_chunks": repair_chunks,
            "repaired_evidence": repaired_evidence,
            "generation": generation,
        }
