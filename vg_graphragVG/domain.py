"""Generic domain helpers used by the VG GraphRAG runtime.

The original project focused on goat-care QA.  These helpers are intentionally
domain-light so the same pipeline can run against HotpotQA without injecting
goat-specific hints.
"""

from __future__ import annotations

import re
from typing import Any


ENTITY_SYNONYMS: dict[str, list[str]] = {
    "united_states": ["us", "u.s.", "usa", "america"],
    "united_kingdom": ["uk", "u.k.", "britain"],
    "film": ["movie", "motion picture"],
    "television": ["tv", "television series"],
    "association_football": ["soccer", "football"],
}


def split_node_aliases(text: str) -> list[str]:
    """Generate simple aliases for an entity label."""
    raw = str(text or "").strip()
    if not raw:
        return []
    lowered = raw.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    snake = re.sub(r"\s+", "_", normalized)
    aliases = [lowered, normalized, snake]
    if "," in raw:
        aliases.append(raw.split(",", 1)[0].strip().lower())
    if "(" in raw:
        aliases.append(raw.split("(", 1)[0].strip().lower())
    return _dedupe([a for a in aliases if a])


def build_domain_hints(
    question: str,
    graph_schema_metadata: dict[str, Any] | None = None,
    disable: bool = False,
    exclude_patterns: list[str] | None = None,
    enable_directqa_linking: bool = False,
) -> dict[str, Any]:
    """Return generic query hints without using dataset answers."""
    if disable:
        return {"alias_terms": [], "diagnostic_focus": [], "matched_patterns": [], "directqa_ids": []}
    q = str(question or "")
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) > 2]
    capitals = re.findall(r"\b[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5}\b", q)
    alias_terms: list[str] = []
    for cap in capitals:
        alias_terms.extend(split_node_aliases(cap))
    for key, vals in ENTITY_SYNONYMS.items():
        if key.replace("_", " ") in q.lower() or any(v in q.lower() for v in vals):
            alias_terms.extend([key] + vals)
    focus = []
    if any(t in tokens for t in {"where", "located", "city", "country", "place"}):
        focus.append("location")
    if any(t in tokens for t in {"who", "person", "actor", "author", "director"}):
        focus.append("person")
    if any(t in tokens for t in {"when", "year", "date"}):
        focus.append("date")
    if any(t in tokens for t in {"same", "both", "larger", "older", "comparison"}):
        focus.append("comparison")
    return {
        "alias_terms": _dedupe(alias_terms + tokens[:12]),
        "diagnostic_focus": _dedupe(focus),
        "matched_patterns": _dedupe(focus),
        "directqa_ids": [] if not enable_directqa_linking else [],
        "graph_run_id": (graph_schema_metadata or {}).get("graph_run_id"),
        "excluded_patterns": list(exclude_patterns or []),
    }


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        v = str(value or "").strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
