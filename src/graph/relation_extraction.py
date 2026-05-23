"""Relation extraction with strict evidence provenance."""

from __future__ import annotations

import json
import re
from typing import Any

from src.graph.canonicalization import canonicalize_relation
from src.graph.entity_extraction import extract_entities_from_text


def extract_relations_from_chunk(chunk: dict[str, Any], max_entities: int = 4, max_edges: int = 8) -> list[dict[str, Any]]:
    """Extract provenance-linked co-mention edges from one chunk."""
    text = chunk.get("text", "")
    title = chunk.get("title", "")
    entities = extract_entities_from_text(text, title=title, max_entities=max_entities)
    edges: list[dict[str, Any]] = []
    if len(entities) < 2:
        return edges
    for i, head in enumerate(entities[:max_entities]):
        for tail in entities[i + 1 : max_entities]:
            if head == tail:
                continue
            edges.append(
                {
                    "head": head,
                    "relation": canonicalize_relation("co_occurs_with"),
                    "tail": tail,
                    "source_doc_id": chunk.get("doc_id"),
                    "source_title": title,
                    "source_chunk_id": chunk.get("chunk_id"),
                    "source_sentence_ids": list(chunk.get("sentence_ids", [])),
                    "supporting_quote": text[:500],
                    "extractor": "rule_based_titlecase_comention_v1",
                    "confidence": 0.6,
                }
            )
            if len(edges) >= max_edges:
                return edges
    return edges


def extract_relations_from_chunk_ollama(
    chunk: dict[str, Any],
    client: Any,
    model_name: str,
    temperature: float = 0.0,
    max_edges: int = 8,
) -> list[dict[str, Any]]:
    """Extract provenance-linked triples from one chunk with Ollama."""
    prompt = (
        "Extract text-grounded triples from the passage. "
        "Return only JSON array items shaped as "
        '{"head":"...","relation":"...","tail":"..."}'
        ". Use short normalized relations and return [] when no clear relation exists.\n\n"
        f"Title: {chunk.get('title')}\n"
        f"Passage: {chunk.get('text')}"
    )
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    if isinstance(response, dict):
        text = str(((response.get("message") or {}).get("content")) or "")
    elif hasattr(response, "message"):
        text = str(getattr(response.message, "content", "") or "")
    else:
        text = str(response)
    match = re.search(r"\[.*\]", text, flags=re.S)
    if not match:
        return []
    try:
        rows = json.loads(match.group(0))
    except Exception:
        return []
    edges: list[dict[str, Any]] = []
    for row in rows[:max_edges]:
        if not isinstance(row, dict):
            continue
        head = str(row.get("head") or "").strip()
        tail = str(row.get("tail") or "").strip()
        relation = canonicalize_relation(str(row.get("relation") or "related_to").strip())
        if not head or not tail or head == tail:
            continue
        edges.append(
            {
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_doc_id": chunk.get("doc_id"),
                "source_title": chunk.get("title"),
                "source_chunk_id": chunk.get("chunk_id"),
                "source_sentence_ids": list(chunk.get("sentence_ids", [])),
                "supporting_quote": str(chunk.get("text") or "")[:500],
                "extractor": "ollama_llm_triple_extraction_v1",
                "confidence": 0.8,
            }
        )
    return edges


def merge_relation_edges(*edge_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate relation edges across extractors while preserving provenance."""
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for edges in edge_lists:
        for edge in edges:
            key = (
                str(edge.get("head") or "").strip().lower(),
                str(edge.get("relation") or "").strip().lower(),
                str(edge.get("tail") or "").strip().lower(),
                str(edge.get("source_chunk_id") or "").strip(),
            )
            if not all(key):
                continue
            if key not in merged or float(edge.get("confidence", 0.0) or 0.0) > float(merged[key].get("confidence", 0.0) or 0.0):
                merged[key] = edge
    return list(merged.values())
