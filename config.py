import os
from pathlib import Path


BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "goat_data_text collection-1.2-eng.txt"
QUESTION_DATASET_PATH = DATA_DIR / "topichop150.csv"
RESULT_DIR = DATA_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


TRIPLE_PROMPT_TEMPLATE = """
You are a domain expert in Animal Husbandry and Veterinary Science, specializing in Goat Management.

Your task is to extract text-grounded knowledge graph triples from the provided text.

CRITICAL RULES:
1. Output ONLY a valid JSON array. No markdown, no explanation, no comments.
2. Each item must be: {{"head": "...", "relation": "...", "tail": "..."}}
3. Return [] if no valid triples are found.
4. Extract only facts that are directly stated or strongly implied by the local context.
5. Do NOT add external knowledge that is not supported by the text.
6. Avoid generic super-nodes such as "goat", "animal", "livestock", or "disease" unless the text explicitly refers to the general category.
7. Prefer specific entities such as lactating_doe, breeding_buck, weaned_kid, newborn_kid, Boer, Saanen, Angora, rumen, hoof, pneumonia, bloat.
8. Use snake_case for common entities and relations.
9. Preserve proper nouns, breed names, drug names, abbreviations, and units when needed.
10. Do not create triples where head, relation, or tail is empty.
11. Do not create triples where head and tail are the same.

ENTITY GUIDELINES:
- Use specific animal roles when possible:
  lactating_doe, pregnant_doe, breeding_buck, weaned_kid, newborn_kid.
- Use specific disease, symptom, treatment, feed, nutrient, anatomy, breed, dosage, and management entities.
- Keep quantitative values as tail entities when important:
  16%_protein, 0.2mg/kg, 8_months.

RELATION GUIDELINES:
Prefer the following relations:

Biological:
- is_breed_of
- used_for
- part_of
- characterized_by
- genetically_predisposed_to

Medical:
- causes
- predisposes_to
- symptoms_include
- symptom_of
- treated_with
- prevents
- diagnosed_by
- transmitted_by

Nutritional:
- contains
- rich_in
- deficient_in
- requires
- limits

Management:
- requires_tool
- scheduled_for
- located_in
- occurs_at
- dosage_is

Use a new relation only if none of the above relations fit.

EXTRACTION STRATEGY:

1. Taxonomy and biology:
Examples:
- (Boer, characterized_by, fast_growth)
- (Saanen, used_for, milk_production)
- (rumen, digests, cellulose)
- (hoof, requires, trimming)

2. Pathology and health:
Extract causal chains when supported by text:
- cause -> disease
- disease -> symptom
- disease -> treatment
- treatment -> disease prevention

Examples:
- (high_humidity, predisposes_to, pneumonia)
- (pneumonia, symptoms_include, coughing)
- (coughing, symptom_of, pneumonia)
- (pneumonia, treated_with, antibiotics)

3. Nutrition and management:
Examples:
- (alfalfa_hay, rich_in, protein)
- (corn, rich_in, energy)
- (pregnant_doe, requires, 16%_protein)
- (ivermectin, dosage_is, 0.2mg/kg)
- (breeding, occurs_at, 8_months)

4. Implicit relations:
You may extract implicit relations only when the implication is clear from the same local context.
Example text: "Feed grain carefully. Too much causes bloat."
Valid triples:
- (excessive_grain, causes, bloat)
- (careful_feeding, prevents, bloat)

OUTPUT FORMAT:
[
  {{"head": "lactating_doe", "relation": "requires", "tail": "calcium"}},
  {{"head": "calcium_deficiency", "relation": "causes", "tail": "milk_fever"}},
  {{"head": "milk_fever", "relation": "treated_with", "tail": "iv_calcium_gluconate"}}
]

Text to extract:
{chunk}
"""


CONFIG = {
    "infrastructure": {
        "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_auth": (
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "neo4jgoat"),
        ),
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "dataset_id": KNOWLEDGE_BASE_PATH.stem.replace(" ", "_") if KNOWLEDGE_BASE_PATH.exists() else "goat_kb_v1",
        "vector_index_name": "chunk_embeddings",
        "fulltext_index_name": "chunk_text_fts",
    },
    "models": {
        "llm_model": "qwen2.5:7b-instruct-fp16",
        "qa_model": "qwen2.5:7b-instruct-fp16",
        "graph_create_model": "deepseekr1-14b-fp16",
        "embed_model": "nomic-embed-text:latest",
        "answer_language": "english",
    },
    "generation": {
        "temperature": 0.0,
        "max_questions": 200,
        "context_window": 4096,
        "batch_size": 10,
        "max_workers": 3,
        "timeout": 150,
        "max_retries": 2,
    },
    "graph_indexing": {
        "triple_retries": 2,
        "write_observation_jsonl": True,
        "observation_log_dir": str(RESULT_DIR / "graph_indexing_runs"),
    },
    "indexing_grid": [
        {"chunk_size": 128, "overlap": 16},
        {"chunk_size": 128, "overlap": 32},
    ],
    "optimal_indexing": {"chunk_size": 512, "overlap": 64},
    "optimization": {
        "hub_threshold_percentile": 95,
        "max_iterations": 1,
        "quality_threshold": 2.5,
        "max_workers": 2,
    },
    "retrieval": {
        "hop_counts": [-1, 0, 1, 2, 3],
        "top_k_values": [5, 10, 15],
        "max_nodes_per_hop": 10,
        "decay_factor": 0.7,
        "enable_reranker": False,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_top_k": 5,
        "reranker_batch_size": 32,
        "reranker_max_length": 512,
        "reranker_device": "cuda",
    },
    "retrieval_grid": {
        "hop_counts": [2],
        "top_k_values": [10],
        "max_questions": 200,
    },
}


GRAPHRAG_KB_PATH = DATA_DIR / "goat_data_text collection-1.2-eng.txt"
GRAPHRAG_QASET1_PATH = DATA_DIR / "QASET1.csv"

GRAPHRAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "source_doc": "goat_kb_v1",
    "run_entity_summary": False,
    "max_workers": 3,
    "embed_model": CONFIG["models"].get("embed_model", "nomic-embed-text:latest"),
    "build_llm_model": "qwen2.5:7b-instruct-fp16",
    "ollama_models": [
        {"name": "qwen2.5:7b-instruct-fp16", "short": "Qwen-7B"},
    ],
    "methods": [
        "LLM-only",
        "VectorSearch",
        "StandardGraphRAG",
        "LocalSearch",
        "GlobalSearch",
    ],
    "category_map": {
        "Production Economics & Farm Management": "Production Econ & Farm Manage",
    },
    "output_prefix": "graphrag_qaset1",
}


ACTIVE_RESEARCH = {
    "active_package": "vg_graphrag (import alias -> vg_graphragVG)",
    "active_graph_run_id": "kg_s2_corpus_plus_directqa_20260511_034322_a4ca36",
    "active_goal": "Beat GraphRAG-hop2 on indirect QA with clean dynamic agentic GraphRAG while keeping no hop2/v5/template leakage.",
    "judge_model": "gpt-5-mini",
    "binary_judge_model": "gpt-5-mini",
    "active_methods": {
        "baselines": ["GraphRAG-hop2", "VectorRAG", "LLM-only"],
        "vg_generation": [
            "run_vg_native_online_ablation.py",
            "run_vg_indirect_ablation_suite.py",
            "run_vg_claim_family_arbitration.py",
            "run_vg_family_purity_ablation.py",
        ],
        "family_registry": [
            "build_family_registry_from_graph_run.py",
        ],
    },
    "results": {
        "vg_claim_family_arbitration_summary": "data/results/vg_claim_family_arbitration_summary.json",
        "vg_family_purity_ablation_summary": "data/results/vg_family_purity_ablation_summary.json",
        "family_induction_summary_cluster_v2": "data/graph_runs/kg_s2_corpus_plus_directqa_20260511_034322_a4ca36/family_induction_summary__pure_graph_cluster_family_v2.json",
    },
    "docs": {
        "startup_report": "docs/PROJECT_STARTUP_REPORT.md",
        "experiment_guide": "docs/EXPERIMENT_GUIDE.md",
        "research_handoff": "docs/RESEARCH_HANDOFF.md",
        "design_notes": "docs/AGENTIC_GRAPHRAG_DESIGN_NOTES.md",
        "project_skill": "PROJECT_SKILL.md",
    },
    "cleanup": {
        "remove_paths": [
            ".pytest_cache",
            "__pycache__",
            "vg_graphraggood",
            "vg_graphrag_true",
            "vg_graphragVG-ClaimFamilyArbitration",
            "vg_graphragvg_claim_family_arbitration_summary",
        ]
    },
}
if os.environ.get("SHOW_CONFIG_BANNER", "").strip() == "1":
    print("CONFIG loaded")
    print(f"  KB       : {KNOWLEDGE_BASE_PATH}")
    print(f"  Questions: {QUESTION_DATASET_PATH}")
    print(f"  Neo4j    : {CONFIG['infrastructure']['neo4j_uri']}")
    print(f"  LLM      : {CONFIG['models']['llm_model']}")
    print(f"  Indexing configs : {len(CONFIG['indexing_grid'])}")
    print(f"  Retrieval configs: {len(CONFIG['retrieval_grid']['hop_counts']) * len(CONFIG['retrieval_grid']['top_k_values'])}")
    print(f"  GraphRAG methods : {len(GRAPHRAG_CONFIG['methods'])}")
