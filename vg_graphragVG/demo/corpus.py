from __future__ import annotations

from vg_graphrag.models import Edge, Node, TextChunk
from vg_graphrag.stores.memory_graph import MemoryGraphStore
from vg_graphrag.stores.memory_text import MemoryTextStore


def build_demo_stores() -> tuple[MemoryGraphStore, MemoryTextStore]:
    nodes = [
        Node("DrugA", "DrugA", aliases=["drug a"]),
        Node("ProteinP", "ProteinP", aliases=["protein p"]),
        Node("DiseaseB", "DiseaseB", aliases=["disease b"]),
        Node("TrialT", "TrialT", aliases=["trial t"]),
        Node("CompanyC", "CompanyC", aliases=["company c"]),
        Node("PathwayX", "PathwayX", aliases=["pathway x"]),
        Node("ProteinQ", "ProteinQ", aliases=["protein q"]),
    ]
    edges = [
        Edge("e1", "DrugA", "ProteinP", "targets", "DrugA targets ProteinP.", "c1"),
        Edge("e2", "ProteinP", "DiseaseB", "associated_with", "ProteinP is associated with DiseaseB.", "c2"),
        Edge("e3", "DrugA", "TrialT", "studied_in", "DrugA was studied in TrialT.", "c3"),
        Edge("e4", "TrialT", "CompanyC", "sponsored_by", "TrialT was sponsored by CompanyC.", "c3"),
        Edge("e5", "ProteinP", "PathwayX", "participates_in", "ProteinP participates in PathwayX.", "c4"),
        Edge("e6", "PathwayX", "DiseaseB", "implicated_in", "PathwayX is implicated in DiseaseB.", "c4"),
    ]
    chunks = [
        TextChunk("c1", "Evidence: DrugA targets ProteinP in biochemical assays."),
        TextChunk("c2", "Evidence: ProteinP is associated with DiseaseB severity."),
        TextChunk("c3", "Evidence: DrugA was studied in TrialT, and TrialT was sponsored by CompanyC."),
        TextChunk("c4", "Evidence: ProteinP participates in PathwayX; PathwayX is implicated in DiseaseB."),
        TextChunk("c5", "Distractor: ProteinQ is unrelated to DrugA."),
        TextChunk("c6", "Weak note: an unverified source suggests DrugA has no effect."),
    ]
    return MemoryGraphStore(nodes, edges, graph_run_id="demo"), MemoryTextStore(chunks)


DEMO_QUESTIONS = [
    "How is DrugA connected to DiseaseB?",
    "Which company is indirectly connected to DrugA through a clinical trial?",
    "Does DrugA affect DiseaseB through a protein target?",
    "What evidence supports the DrugA to DiseaseB connection?",
]
