"""VG-compatible adapters that bridge HotpotQA retrievers into VG abstractions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vg_graphrag.models import AgentState as VGAgentState
from vg_graphrag.models import QueryAnalysis, RunConfig, ToolResult
from vg_graphrag.pipeline.analyzer import analyze_query
from vg_graphrag.pipeline.executor import execute_plan
from vg_graphrag.pipeline.planner import create_dual_channel_plan
from vg_graphrag.runtime_skill import load_runtime_skill


def inspect_vg_environment(root: Path) -> dict[str, Any]:
    """Inspect which VG GraphRAG modules are available for HotpotQA integration."""
    checked = {
        "skill_md": root / "vg_graphragVG" / "SKILL.md",
        "pipeline_runner": root / "vg_graphragVG" / "pipeline" / "runner.py",
        "pipeline_planner": root / "vg_graphragVG" / "pipeline" / "planner.py",
        "pipeline_executor": root / "vg_graphragVG" / "pipeline" / "executor.py",
        "tools_init": root / "vg_graphragVG" / "tools" / "__init__.py",
        "adapters_graph_run_loader": root / "vg_graphragVG" / "adapters" / "graph_run_loader.py",
    }
    exists = {name: path.exists() for name, path in checked.items()}
    importable_modules: list[str] = []
    for module_name in [
        "vg_graphrag.pipeline.analyzer",
        "vg_graphrag.pipeline.planner",
        "vg_graphrag.pipeline.executor",
        "vg_graphrag.runtime_skill",
        "vg_graphrag.models",
    ]:
        try:
            __import__(module_name)
            importable_modules.append(module_name)
        except Exception:
            continue
    return {
        "exists": exists,
        "importable_modules": importable_modules,
        "integration_possible": len(importable_modules) >= 4,
    }


@dataclass
class VGExecutionBundle:
    """Structured result from the VG planner/executor pass."""

    analysis: QueryAnalysis
    vg_state: VGAgentState
    config: RunConfig
    modules_called: list[str]
    adapters_used: list[str]


class _BaseAdapter:
    """Base class for VG tool adapters."""

    name: str

    def __init__(self, hotpot_backend: Any):
        self.backend = hotpot_backend

    def _result(self, query: dict[str, Any], rows: list[dict[str, Any]], elapsed_ms: int) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            query=query,
            results=rows,
            cost_metadata={"latency_ms": elapsed_ms},
            provenance_metadata={"adapter_class": type(self).__name__},
        )


class VGTextSearchToolAdapter(_BaseAdapter):
    """Map VG TextSearch to the Hotpot text retriever."""

    name = "TextSearch"

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        rows = self.backend.retrieve(str(input.get("query", "")), top_k=int(input.get("limit", 10)))
        return self._result(input, rows, int((time.perf_counter() - started) * 1000))


class VGClaimSearchToolAdapter(_BaseAdapter):
    """Compatibility adapter for ClaimSearch using text retrieval over the same corpus."""

    name = "ClaimSearch"

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        rows = self.backend.retrieve(str(input.get("query", "")), top_k=int(input.get("limit", 10)))
        for row in rows:
            row["source"] = "vg_claim_search_adapter"
        return self._result(input, rows, int((time.perf_counter() - started) * 1000))


class VGEntitySearchToolAdapter(_BaseAdapter):
    """Map VG EntitySearch to the Hotpot graph store entity linker."""

    name = "EntitySearch"

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        rows = self.backend.store.entity_search(str(input.get("query", "")), limit=int(input.get("limit", 8)))
        return self._result(input, rows, int((time.perf_counter() - started) * 1000))


class VGGraphNeighborToolAdapter(_BaseAdapter):
    """Map VG GraphNeighbor to one-hop graph retrieval."""

    name = "GraphNeighbor"

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        query = " ".join(str(x) for x in input.get("node_ids", []) if x) or str(input.get("node_query", ""))
        result = self.backend.retrieve(
            query,
            depth=min(int(input.get("max_hops", 1)), 1),
            top_k_paths=max(1, int(input.get("limit", 5) or 5)),
            max_nodes_per_hop=max(1, int(input.get("max_hops", 1) or 1) * 5),
        )
        rows = list(result.get("retrieved_edges", []))
        tool_result = self._result(input, rows, int((time.perf_counter() - started) * 1000))
        tool_result.provenance_metadata["graph_relation_type"] = "neighbor"
        return tool_result


class VGPathSearchToolAdapter(_BaseAdapter):
    """Map VG PathSearch to bounded multi-hop Hotpot graph retrieval."""

    name = "PathSearch"

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        query = " ".join(
            [
                str(input.get("source_query", "") or ""),
                str(input.get("target_query", "") or ""),
                " ".join(str(x) for x in input.get("source_ids", []) if x),
                " ".join(str(x) for x in input.get("target_ids", []) if x),
            ]
        ).strip()
        result = self.backend.retrieve(
            query,
            depth=min(int(input.get("max_hops", 2)), 2),
            top_k_paths=max(1, int(input.get("limit", 5) or 5)),
            max_nodes_per_hop=10,
        )
        rows = list(result.get("retrieved_paths", []))
        tool_result = self._result(input, rows, int((time.perf_counter() - started) * 1000))
        tool_result.provenance_metadata["graph_relation_type"] = "path"
        return tool_result


class VGHybridSearchToolAdapter(_BaseAdapter):
    """Map VG HybridSearch to text retrieval with graph-linked provenance markers."""

    name = "HybridSearch"

    def __init__(self, text_backend: Any, graph_backend: Any):
        super().__init__(text_backend)
        self.graph_backend = graph_backend

    def run(self, input: dict[str, Any], state: VGAgentState) -> ToolResult:
        started = time.perf_counter()
        text_rows = self.backend.retrieve(str(input.get("query", "")), top_k=int(input.get("limit", 10)))
        graph_rows = self.graph_backend.retrieve(
            str(input.get("query", "")),
            depth=min(int(input.get("max_hops", 2)), 2),
            top_k_paths=max(1, int(input.get("limit", 5) or 5)),
            max_nodes_per_hop=10,
        ).get("graph_evidence_chunks", [])
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in list(text_rows) + list(graph_rows):
            cid = str(row.get("chunk_id") or "")
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            merged.append(row)
        return self._result(input, merged, int((time.perf_counter() - started) * 1000))


def run_vg_planner_pass(
    question: str,
    question_id: str,
    text_retriever: Any,
    graph_retriever: Any,
    skill_profile_path: str | None,
    graph_run_id: str,
    max_tool_calls: int,
    max_hops: int,
    max_chunks: int,
) -> VGExecutionBundle:
    """Execute the reusable VG analyzer/planner/executor stack via HotpotQA adapters."""
    runtime_skill = load_runtime_skill(skill_profile_path)
    metadata = {
        "runtime_skill": runtime_skill,
        "graph_run_id": graph_run_id,
        "question_id": question_id,
        "evaluation_mode": True,
        "generic_agentic_only": True,
        "graph_reliance_mode": "graph_evidence_allowed",
    }
    analysis = analyze_query(question, metadata)
    config = RunConfig(
        max_tool_calls=max_tool_calls,
        max_hops=max_hops,
        max_chunks=max_chunks,
        use_graph_tools=True,
        use_text_tools=True,
        use_verifier=False,
        use_refinement_loop=False,
        graph_run_id=graph_run_id,
    )
    vg_state = VGAgentState(question=question, iterations=0)
    plan = create_dual_channel_plan(
        subquery=type("SubQueryLike", (), {"subquery_id": question_id, "text": question, "grounded_text": question})(),
        analysis=analysis,
        state=vg_state,
        config=config,
    )
    tools = {
        "TextSearch": VGTextSearchToolAdapter(text_retriever),
        "ClaimSearch": VGClaimSearchToolAdapter(text_retriever),
        "EntitySearch": VGEntitySearchToolAdapter(graph_retriever),
        "GraphNeighbor": VGGraphNeighborToolAdapter(graph_retriever),
        "PathSearch": VGPathSearchToolAdapter(graph_retriever),
        "HybridSearch": VGHybridSearchToolAdapter(text_retriever, graph_retriever),
    }
    execute_plan(plan, tools, vg_state, config)
    return VGExecutionBundle(
        analysis=analysis,
        vg_state=vg_state,
        config=config,
        modules_called=[
            "vg_graphrag.runtime_skill.load_runtime_skill",
            "vg_graphrag.pipeline.analyzer.analyze_query",
            "vg_graphrag.pipeline.planner.create_dual_channel_plan",
            "vg_graphrag.pipeline.executor.execute_plan",
        ],
        adapters_used=[
            "VGTextSearchToolAdapter",
            "VGClaimSearchToolAdapter",
            "VGEntitySearchToolAdapter",
            "VGGraphNeighborToolAdapter",
            "VGPathSearchToolAdapter",
            "VGHybridSearchToolAdapter",
        ],
    )
