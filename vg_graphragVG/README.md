# VG-GraphRAG

VG-GraphRAG v2 is an independent dynamic retrieval-time Agentic GraphRAG system implemented in a GraphSearch-style workflow. It starts from the raw question, decomposes it into retrieval subqueries, runs semantic and relational retrieval channels for each subquery, refines retrieved context, verifies evidence sufficiency, expands only missing evidence, and synthesizes a native answer grounded only in VG-retrieved evidence.

GraphRAG-hop2 is not part of the VG runtime loop. v5 / EVAG-RAG is not part of this version. VG-GraphRAG does not generate patches over a frozen base answer. VG-GraphRAG does not call v5 strict gate. VG-GraphRAG does not call `judge_openai.py`.

## Primary Mode: `vg_native_answer`

In primary `vg_native_answer` mode, VG-GraphRAG starts from the raw question and performs independent dynamic retrieval over graph/text stores. It builds an EvidencePackage, verifies evidence sufficiency, refines retrieval when necessary, and synthesizes an answer grounded only in VG-retrieved evidence. GraphRAG-hop2 and v5 are not required and are not used in this version.

The v2 runtime loop is:

1. Query Analyzer
2. Query Decomposition (QD)
3. Query Grounding (QG)
4. Dual-channel retrieval
5. Context Refinement (CR)
6. Logic Drafting (LD)
7. Evidence Verification (EV)
8. Query Expansion (QE)
9. VG-native Answer Synthesizer or abstention

## Positioning

VG-GraphRAG is:

- independent dynamic retrieval-time Agentic GraphRAG
- GraphSearch-style and Graph-CoT-inspired through reasoning -> graph interaction -> graph/text execution
- verifier-guided
- non-RL
- hybrid graph/text evidence retrieval
- capable of `vg_native_answer` without hop2 or v5

VG-GraphRAG is not:

- GraphRAG-hop2 post-processing
- v5 patch-only repair
- EVAG-RAG
- Graph Counselor
- reinforcement learning
- an answer-score judge
- a chat UI

## Architecture

- `adapters/`: graph_run artifact loader and optional Neo4j adapter.
- `stores/`: graph/text protocols and deterministic in-memory stores.
- `tools/`: EntitySearch, GraphNeighbor, PathSearch, TextSearch, HybridSearch.
- `pipeline/`: analyzer, decomposition, grounding, planner, executor, context refinement, evidence builder, logic drafting, verifier, query expansion, synthesizer, runner.
- `demo/`: offline mini corpus for tests and CLI smoke checks.

Deprecated compatibility files such as `adapters/v5_gate_adapter.py` and `pipeline/patcher.py` may remain in the repository for historical compatibility, but they are not imported or used by the active `vg_native_answer` path.

## CLI

```powershell
python -m vg_graphrag --demo
python -m vg_graphrag "How is DrugA connected to DiseaseB?"
python -m vg_graphrag --graph-run-id kg_s2_corpus_plus_directqa_20260511_034322_a4ca36 --case-scope triggered
```

Diagnostic graph-run outputs are written to:

- `data/results/vg_graphrag_diagnostic_cases.jsonl`
- `data/results/vg_graphrag_tool_traces.jsonl`
- `data/results/vg_graphrag_evidence_packages.jsonl`
- `data/results/vg_graphrag_verifier_reports.jsonl`
- `data/results/vg_graphrag_native_answers.jsonl`
- `data/results/vg_graphrag_summary.json`
- `data/results/vg_graphrag_report.md`
- `data/results/vg_graphrag_next_step.md`

The diagnostic command does not call an answer judge and does not overwrite existing experiment result files.

## GraphSearch-Style State

Each case records:

- `subqueries`: QD output with optional relational triple hints.
- `channel_evidence`: per-subquery semantic evidence and relational evidence.
- `logic_draft`: LD reasoning steps and evidence gaps.
- `verifier_summary.missing_evidence_map`: EV output keyed by subquery id.
- `tool_trace`: dynamic retrieval actions, context refinement actions, and verification actions.

Pseudo-code:

```python
def vg_graphsearch_v2(question, graph, text, config):
    analysis = analyze_query(question)
    subqueries = decompose_query(question, analysis)
    subqueries = decompose_relational_queries(subqueries, analysis)

    state = AgentState(question=question, analysis=analysis)
    for iteration in range(config.max_iterations):
        for subquery in subqueries:
            grounded = ground_subquery(subquery, state.channel_evidence)
            plan = create_dual_channel_plan(grounded, analysis, state, config)
            results = execute_plan(plan, tools, state, config)
            refined = refine_context(grounded.subquery_id, grounded.grounded_text, results)
            state.channel_evidence.append(refined)

        draft = draft_logic(question, state.channel_evidence)
        report = verify_graphsearch_evidence(question, analysis, state.channel_evidence, draft)
        if report.verdict == "accept":
            return synthesize_vg_native_answer(question, state, report)
        if report.verdict == "abstain":
            return synthesize_abstention(question, state, report)
        subqueries = expand_queries(question, state.channel_evidence, report)

    return synthesize_best_effort_or_abstention(question, state, report)
```

The semantic channel uses `TextSearch` and chunk/claim text. The relational channel uses `EntitySearch`, `GraphNeighbor`, `PathSearch`, and graph-linked `HybridSearch`. GraphRAG-hop2 and v5 are not part of either channel.

## Tests

```powershell
pytest tests/test_vg_*.py
```

Tests use only the in-memory demo corpus. They do not require Neo4j, Qdrant, OpenAI, external API keys, or network access.

## Ablation Switches

`RunConfig` exposes:

- `use_verifier`
- `use_graph_tools`
- `use_text_tools`
- `use_refinement_loop`
- `fixed_k_hop_only`
- `max_iterations`
- `max_tool_calls`
- `max_hops`
- `max_chunks`

Useful ablations:

- without verifier
- without graph tools
- without text tools
- without refinement loop
- fixed k-hop only
- hybrid no-verifier baseline
- VG-native retrieval-only diagnostic
