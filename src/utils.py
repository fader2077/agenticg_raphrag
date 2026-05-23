"""Minimal legacy utility functions used by builder.py."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_think_tokens(text: str) -> str:
    """Remove common think-token wrappers."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    return re.sub(r"<think>.*$", "", text, flags=re.S).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Token-count chunk text for legacy graph builder compatibility."""
    words = str(text or "").split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)]
