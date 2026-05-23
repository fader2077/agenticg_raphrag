"""Entity and relation canonicalization utilities."""

from __future__ import annotations

import re


def canonicalize_entity(text: str) -> str:
    """Canonicalize an entity surface form while keeping readable labels."""
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    cleaned = cleaned.strip(".,;:()[]{}\"'")
    return cleaned


def entity_key(text: str) -> str:
    """Return a stable lower-case entity key."""
    text = canonicalize_entity(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonicalize_relation(text: str) -> str:
    """Canonicalize a relation label to snake_case."""
    rel = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "related_to").lower())
    return re.sub(r"_+", "_", rel).strip("_") or "related_to"
