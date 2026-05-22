from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Tuple

from vg_graphrag.domain import ENTITY_SYNONYMS, split_node_aliases
from vg_graphrag.models import Edge, Node, TextChunk
from vg_graphrag.stores.memory_graph import MemoryGraphStore
from vg_graphrag.stores.memory_text import MemoryTextStore


def _read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def _first(r: dict, keys: list[str], default: str = "") -> str:
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            return str(v)
    return default


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _augment_aliases(name: str, provenance: dict) -> list[str]:
    aliases = split_node_aliases(name)
    norm = str(provenance.get("name_norm") or "")
    if norm:
        aliases.extend(split_node_aliases(norm))
    for canonical, syns in ENTITY_SYNONYMS.items():
        if canonical in norm or canonical in name.lower():
            aliases.extend(syns)
    return _dedupe([a.lower() for a in aliases])


def _relation_aliases(rel: str) -> list[str]:
    rel_l = (rel or "").lower()
    aliases = [rel_l]
    if rel_l == "symptoms_include":
        aliases.append("symptom_of")
    if rel_l == "treated_with":
        aliases.append("treats")
    if rel_l == "prevents":
        aliases.append("prevented_by")
    if rel_l == "characterized_by":
        aliases.append("associated_with")
    if rel_l == "indicates":
        aliases.append("associated_with")
    return _dedupe(aliases)


def load_graph_run(graph_run_id: str | None = None, graph_run_dir: str | Path | None = None) -> Tuple[MemoryGraphStore, MemoryTextStore, dict]:
    if graph_run_dir:
        root = Path(graph_run_dir)
    elif graph_run_id:
        root = Path("data/graph_runs") / graph_run_id
    else:
        candidates = sorted(Path("data/graph_runs").glob("kg_s2_corpus_plus_directqa_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        non_empty = [p for p in candidates if (p / "relations.jsonl").exists() and (p / "relations.jsonl").stat().st_size > 0]
        root = non_empty[0] if non_empty else candidates[0]
    manifest = {}
    if (root / "run_manifest.json").exists():
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    gid = manifest.get("graph_run_id") or root.name

    nodes = []
    node_ids = set()
    for r in _read_jsonl(root / "entities.jsonl"):
        node_id = _first(r, ["entity_id", "node_id", "id", "name", "entity"])
        name = _first(r, ["name", "canonical_name", "entity", "label"], node_id)
        if node_id:
            aliases = r.get("aliases", []) if isinstance(r.get("aliases"), list) else []
            node = Node(
                node_id,
                name,
                aliases=_dedupe([str(a).lower() for a in aliases] + _augment_aliases(name, r)),
                node_type=_first(r, ["type", "entity_type"], "entity"),
                provenance=r,
            )
            nodes.append(node)
            node_ids.add(node_id)

    chunks = []
    for r in _read_jsonl(root / "chunks.jsonl"):
        cid = _first(r, ["chunk_id", "id"])
        txt = _first(r, ["text", "content"])
        if cid and txt:
            chunks.append(TextChunk(cid, txt, _first(r, ["source_document_id", "document_id"], None), r))

    edges = []
    edge_keys = set()
    for i, r in enumerate(_read_jsonl(root / "relations.jsonl")):
        source = _first(r, ["source", "head", "source_entity", "from", "entity_1"])
        target = _first(r, ["target", "tail", "target_entity", "to", "entity_2"])
        rel = _first(r, ["relation", "type", "predicate"], "related_to")
        if not source or not target:
            continue
        eid = _first(r, ["relation_id", "edge_id", "id"], f"r{i}")
        edge = Edge(eid, source, target, rel, _first(r, ["supporting_quote", "quote", "evidence"], ""), _first(r, ["source_chunk_id", "chunk_id"], None), _first(r, ["source_document_id", "document_id"], None), r.get("confidence"), r)
        edges.append(edge)
        edge_keys.add((source, rel, target))

    claims = list(_read_jsonl(root / "claims.jsonl"))
    for i, claim in enumerate(claims):
        head = _first(claim, ["head", "source", "entity_1"])
        tail = _first(claim, ["tail", "target", "entity_2"])
        rel = _first(claim, ["relation", "type", "predicate"], "related_to")
        claim_text = _first(claim, ["claim_text", "text"], "")
        if not head or not tail:
            continue
        for nid in (head, tail):
            if nid not in node_ids:
                node = Node(nid, nid, aliases=_augment_aliases(nid, claim), node_type="entity", provenance=claim)
                nodes.append(node)
                node_ids.add(nid)
        if (head, rel, tail) not in edge_keys:
            edges.append(Edge(f"claim_edge::{i}", head, tail, rel, _first(claim, ["supporting_quote"], ""), None, None, claim.get("confidence"), claim))
            edge_keys.add((head, rel, tail))
        for rel_alias in _relation_aliases(rel):
            if rel_alias == rel or (head, rel_alias, tail) in edge_keys:
                continue
            edges.append(Edge(f"claim_alias::{i}::{rel_alias}", head, tail, rel_alias, _first(claim, ["supporting_quote"], ""), None, None, claim.get("confidence"), claim))
            edge_keys.add((head, rel_alias, tail))
        claim_node_id = f"claim::{i}"
        nodes.append(
            Node(
                claim_node_id,
                claim_text or f"{head} {rel} {tail}",
                aliases=_dedupe(split_node_aliases(claim_text) + split_node_aliases(head) + split_node_aliases(tail)),
                node_type="claim",
                provenance=claim,
            )
        )
        edges.append(Edge(f"claim_subject::{i}", head, claim_node_id, "supports_claim", _first(claim, ["supporting_quote"], ""), None, None, claim.get("confidence"), claim))
        edges.append(Edge(f"claim_target::{i}", claim_node_id, tail, "claim_targets", _first(claim, ["supporting_quote"], ""), None, None, claim.get("confidence"), claim))

    return MemoryGraphStore(nodes, edges, graph_run_id=gid), MemoryTextStore(chunks), {"graph_run_id": gid, "artifact_dir": str(root), "manifest": manifest, "node_count": len(nodes), "edge_count": len(edges), "chunk_count": len(chunks)}
