# VG-GraphRAG Skill

This skill defines the intended operating contract for `vg_native_answer` mode.
`SKILL.md` remains the human-readable contract, while the runtime now loads a structured skill profile from:

- `vg_graphrag/domains/goat/skill_profile.json`

The pipeline is therefore skill-driven at runtime through structured config, not by parsing this markdown file directly.

## Scope

- Applies to `vg_graphrag` runtime only.
- Does not use GraphRAG-hop2 cached answers.
- Does not use v5 / EVAG-RAG gates.
- Does not call answer judge during retrieval runtime.

## Required Runtime Loop

1. Query Analyzer
2. Query Decomposition (QD)
3. Query Grounding (QG)
4. Dual-channel Tool Execution
5. Context Refinement (CR)
6. Logic Drafting (LD)
7. Evidence Verification (EV)
8. Query Expansion (QE, if needed)
9. VG-native Answer Synthesis (or abstain)

## Tool-Use Policy

- Always execute at least one dynamic retrieval tool for non-abstained outputs.
- Every subquery should record both semantic evidence and relational evidence.
- Semantic channel: `TextSearch` over chunks/claims.
- Relational channel: `EntitySearch`, `GraphNeighbor`, `PathSearch`, and graph-linked `HybridSearch`.
- For multi-hop/evidence-demanding questions, include one graph tool and one text tool.
- Keep traversal bounded by `RunConfig.max_hops`.

## Modules vs Tools

- Tools are retrieval primitives that touch graph/text state:
  - `EntitySearch`
  - `ClaimSearch`
  - `TextSearch`
  - `GraphNeighbor`
  - `PathSearch`
  - `HybridSearch`
- Modules are controller stages:
  - Query Analyzer
  - Query Decomposition
  - Query Grounding
  - Context Refinement
  - Logic Drafting
  - Evidence Verification
  - Query Expansion
  - Self Reflection
  - Native Answer Synthesis
- Modules should decide how to use tools; tools should not own the high-level search policy.
- Runtime skill profile can bias tool ordering and reflection-triggered tool preference, but the controller still lives in pipeline code.

## Verifier Policy

- `accept` when evidence is sufficient and critical failures are absent.
- `refine` when evidence is partial and iteration budget remains.
- `abstain` when evidence remains insufficient at max iterations.
- EV must write `missing_evidence_map` keyed by subquery id.
- QE must generate follow-up subqueries only from `missing_evidence_map`.
- Never use reference answers or judge scores at runtime.

## Answer Synthesis Policy

- Synthesize from retrieved evidence only.
- Prefer grounded sentence extraction from retrieved chunks.
- If graph path exists, include path relation in concise form.
- Avoid raw "Question:/Answer:" wrappers from retrieved QA chunks.
- If insufficient evidence, abstain and list missing information.

## Mandatory Output Flags

Every case must include:

- `vg_mode = "vg_native_answer"`
- `independent_dynamic_retrieval`
- `dynamic_tool_call_count`
- `used_hop2_context_ids = false`
- `used_hop2_answer_as_input = false`
- `used_v5_gate = false`
- `used_v5_outputs = false`
- `hop2_usage = "none"`
- `v5_usage = "none"`
- `invalid_vg_reason`

## Operational Checklist

- Verify active graph state before online evaluation.
- Use S2 graph run for non-oracle evaluation.
- Separate generation and judging stages.
- Use `judge_openai.py` and `judge_binary_correctness.py` only after generation.
- Keep direct (1-102) and indirect (103-158) metrics separated.
