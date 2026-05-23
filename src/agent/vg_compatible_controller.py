"""Skill-driven HotpotQA AgenticGraphRAG controller with VG abstraction integration."""

from __future__ import annotations

from typing import Any

from src.agent.skills import (
    EvidenceFusionSkill,
    GraphRetrievalSkill,
    GroundedAnsweringSkill,
    QuestionAnalysisSkill,
    RepairSkill,
    TextRetrievalSkill,
    VectorRetrievalSkill,
)
from src.agent.state import AgentState
from src.agent.tools import call_tool
from src.agent.verifier import verify_answer
from src.generation.answer_generator import AnswerGenerator


class AgenticGraphRAGController:
    """VG-compatible bounded deterministic controller with explicit skill stages."""

    def __init__(self, text_retriever: Any, vector_retriever: Any, graph_retriever: Any, config: dict[str, Any]):
        self.text_retriever = text_retriever
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.config = config
        generation_cfg = config.get("generation", {})
        self.generator = AnswerGenerator(
            temperature=float(generation_cfg.get("temperature", 0.0)),
            max_context_tokens=int(generation_cfg.get("max_context_tokens", 6000)),
            provider=str(generation_cfg.get("provider", "ollama")),
            qa_model=generation_cfg.get("qa_model"),
            ollama_host=generation_cfg.get("ollama_host"),
            deterministic_fallback_enabled=bool(generation_cfg.get("deterministic_fallback_enabled", True)),
            use_deterministic_for_main_eval=bool(generation_cfg.get("use_deterministic_for_main_eval", False)),
        )
        retrieval_cfg = config.get("retrieval", {})
        agent_cfg = config.get("agent", {})
        self.question_analysis_skill = QuestionAnalysisSkill(text_retriever, graph_retriever, config)
        self.text_skill = TextRetrievalSkill(text_retriever, retrieval_cfg)
        self.vector_skill = VectorRetrievalSkill(vector_retriever, retrieval_cfg)
        self.graph_skill = GraphRetrievalSkill(graph_retriever, retrieval_cfg, agent_cfg)
        self.fusion_skill = EvidenceFusionSkill()
        self.answering_skill = GroundedAnsweringSkill(self.generator)
        self.repair_skill = RepairSkill(text_retriever, self.answering_skill, retrieval_cfg)

    def run(self, qid: str, question: str, ablation: str = "full") -> dict[str, Any]:
        """Execute the skill-driven AgenticGraphRAG pipeline for one question."""
        state = AgentState(qid=qid, question=question)
        state.pipeline_version = "hotpotqa_agentic_graphrag_v1"
        state.tools_used = [
            "vg_analyze_query",
            "vg_create_dual_channel_plan",
            "vg_execute_plan",
            "text_search",
            "vector_search",
            "graph_expand",
            "evidence_fuse",
            "answer_generate",
            "evidence_verify",
            "repair_retrieve",
        ]
        state.skills_used = [
            "QuestionAnalysisSkill",
            "TextRetrievalSkill",
            "VectorRetrievalSkill",
            "GraphRetrievalSkill",
            "EvidenceFusionSkill",
            "GroundedAnsweringSkill",
            "RepairSkill",
        ]
        state.vg_graphrag_integration = {
            "vg_graphrag_reference_used": False,
            "vg_graphrag_modules_called": [],
            "vg_graphrag_adapters_used": [],
            "vg_graphrag_abstractions_embedded": list(state.skills_used),
            "controller_backend": "vg_compatible_bounded_deterministic_v1",
            "integration_status": "adapter_integrated",
        }
        agent_cfg = self.config.get("agent", {})

        analysis = call_tool(state.tool_trace, "question_analysis_skill", self.question_analysis_skill.run, question, state)
        state.analysis = analysis
        vg_bundle = analysis.get("vg_bundle")
        if vg_bundle is not None:
            state.vg_graphrag_integration["vg_graphrag_reference_used"] = True
            state.vg_graphrag_integration["vg_graphrag_modules_called"] = list(vg_bundle.modules_called)
            state.vg_graphrag_integration["vg_graphrag_adapters_used"] = list(vg_bundle.adapters_used)
            state.vg_graphrag_integration["integration_status"] = "native_vg_modules_called"
            for item in getattr(vg_bundle.vg_state, "tool_trace", []):
                state.tool_trace.append(
                    {
                        "tool": item.get("tool_name"),
                        "input": item.get("input"),
                        "status": "ok" if not item.get("errors") else "error",
                        "num_results": item.get("result_count", 0),
                        "latency_ms": None,
                        "module_path": "vg_graphrag.pipeline.executor.execute_plan",
                        "adapter_class": item.get("graph_relation_type") or "",
                    }
                )

        text_chunks: list[dict[str, Any]] = []
        vector_chunks: list[dict[str, Any]] = []
        graph_result = {"retrieved_entities": [], "retrieved_paths": [], "retrieved_edges": [], "graph_evidence_chunks": [], "seed_entities": []}
        if ablation != "no_text_route":
            text_chunks = call_tool(state.tool_trace, "text_retrieval_skill", self.text_skill.run, question, state)
        if ablation != "no_vector_route":
            vector_chunks = call_tool(state.tool_trace, "vector_retrieval_skill", self.vector_skill.run, question, state)
        if ablation != "no_graph_route":
            graph_result = call_tool(state.tool_trace, "graph_retrieval_skill", self.graph_skill.run, question, state)
        graph_chunks = graph_result.get("graph_evidence_chunks", [])
        fused = call_tool(state.tool_trace, "evidence_fusion_skill", self.fusion_skill.run, text_chunks, vector_chunks, graph_chunks, state)
        generated = call_tool(state.tool_trace, "grounded_answering_skill", self.answering_skill.run, question, fused, state)
        verifier = {"verdict": "not_run", "unsupported_claims": [], "required_missing_evidence": [], "trace": {}}
        verifier_enabled = bool(agent_cfg.get("verifier_enabled", True)) and ablation != "no_verifier"
        if verifier_enabled:
            verifier = call_tool(state.tool_trace, "evidence_verify", verify_answer, question, generated["answer"], fused)
            state.verifier_trace.append({"round": 0, **verifier})
        repaired = False
        if (
            verifier_enabled
            and ablation != "no_repair"
            and bool(agent_cfg.get("repair_enabled", True))
            and verifier.get("verdict") in {"unsupported", "partially_supported", "insufficient_evidence"}
            and int(agent_cfg.get("max_repair_rounds", 1)) > 0
        ):
            repaired = True
            state.repair_rounds += 1
            repaired_payload = call_tool(
                state.tool_trace,
                "repair_skill",
                self.repair_skill.run,
                question,
                generated["answer"],
                fused,
                state,
            )
            state.repair_trace.append(
                {
                    "round": state.repair_rounds,
                    "reason": verifier.get("verdict"),
                    "repair_query": repaired_payload.get("repair_query"),
                    "repair_chunk_count": len(repaired_payload.get("repair_chunks", [])),
                }
            )
            fused = repaired_payload["repaired_evidence"]
            generated = repaired_payload["generation"]
            verifier = call_tool(state.tool_trace, "evidence_verify", verify_answer, question, generated["answer"], fused)
            state.verifier_trace.append({"round": state.repair_rounds, **verifier})
        state.retrieved_chunks = fused
        state.retrieved_entities = list(graph_result.get("retrieved_entities", []))
        state.retrieved_paths = list(graph_result.get("retrieved_paths", []))
        state.verifier = verifier
        return {
            "pred_answer": generated["answer"],
            "retrieved_chunks": fused,
            "retrieved_entities": state.retrieved_entities,
            "retrieved_edges": graph_result.get("retrieved_edges", []),
            "retrieved_paths": state.retrieved_paths,
            "seed_entities": graph_result.get("seed_entities", []),
            "tool_trace": state.tool_trace,
            "verifier": verifier,
            "repair_used": repaired,
            "repair_trace": state.repair_trace,
            "verifier_trace": state.verifier_trace,
            "pipeline_version": state.pipeline_version,
            "tools_used": state.tools_used,
            "skills_used": state.skills_used,
            "vg_graphrag_integration": state.vg_graphrag_integration,
            "generation": generated,
        }
