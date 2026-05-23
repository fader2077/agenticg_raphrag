"""Lightweight entity extraction and question entity linking for HotpotQA."""

from __future__ import annotations

import re
from typing import Iterable

from src.graph.canonicalization import canonicalize_entity


STOP_ENTITIES = {
    "The",
    "This",
    "That",
    "These",
    "There",
    "It",
    "He",
    "She",
    "They",
    "What",
    "Which",
    "Who",
    "Where",
    "When",
    "Were",
    "Was",
    "Did",
    "Do",
    "Does",
    "Is",
    "Are",
}


ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5}\b")


def extract_entities_from_text(text: str, title: str | None = None, max_entities: int = 12) -> list[str]:
    """Extract title-case, quoted, and title-derived entity mentions."""
    seen: set[str] = set()
    entities: list[str] = []
    scan_text = str(text or "")[:1200]
    for value in [title or ""]:
        clean = canonicalize_entity(value)
        if clean and clean not in seen:
            seen.add(clean)
            entities.append(clean)
    for quoted in re.findall(r'"([^"]{2,120})"', scan_text):
        clean = canonicalize_entity(quoted)
        if clean and clean not in seen:
            seen.add(clean)
            entities.append(clean)
    for match in ENTITY_RE.findall(scan_text):
        clean = canonicalize_entity(match)
        if not clean or clean in STOP_ENTITIES or len(clean) < 2:
            continue
        if clean not in seen:
            seen.add(clean)
            entities.append(clean)
        if len(entities) >= max_entities:
            break
    return entities[:max_entities]


def link_question_entities(question: str, graph_entities: Iterable[str], max_entities: int = 8) -> list[str]:
    """Link question mentions to graph entity labels with exact and token-overlap matching."""
    graph_list = list(graph_entities)
    by_lower = {g.lower(): g for g in graph_list}
    seeds: list[str] = []
    for ent in extract_entities_from_text(question, max_entities=max_entities):
        if ent.lower() in by_lower and by_lower[ent.lower()] not in seeds:
            seeds.append(by_lower[ent.lower()])
        elif ent not in seeds:
            seeds.append(ent)
    q_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", question) if len(t) > 2}
    scored: list[tuple[float, str]] = []
    for ent in graph_list:
        toks = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", ent) if len(t) > 2}
        if not toks:
            continue
        overlap = len(q_tokens & toks) / len(toks)
        if overlap > 0:
            scored.append((overlap, ent))
    for _, ent in sorted(scored, reverse=True):
        if ent not in seeds:
            seeds.append(ent)
        if len(seeds) >= max_entities:
            break
    return seeds[:max_entities]
