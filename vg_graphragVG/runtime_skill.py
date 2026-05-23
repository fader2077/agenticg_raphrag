"""Small runtime-skill compatibility layer for VG GraphRAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_runtime_skill(path: str | None = None) -> dict[str, Any]:
    """Load an optional runtime skill profile."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def skill_flag(skill: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read a nested skill flag with a default."""
    sec = skill.get(section, {}) if isinstance(skill, dict) else {}
    if isinstance(sec, dict):
        return sec.get(key, default)
    return default


def pattern_term_score(
    skill: dict[str, Any],
    matched_patterns: set[str] | list[str],
    text: str,
    default_boost: float = 1.0,
    default_penalty: float = 0.0,
) -> float:
    """Score text using optional pattern boost/penalty terms."""
    if not skill:
        return 0.0
    text_l = str(text or "").lower()
    score = 0.0
    patterns = skill.get("patterns", {}) if isinstance(skill, dict) else {}
    for pattern in matched_patterns or []:
        row = patterns.get(pattern, {}) if isinstance(patterns, dict) else {}
        for term in row.get("boost_terms", []) if isinstance(row, dict) else []:
            if str(term).lower() in text_l:
                score += float(row.get("boost", default_boost))
        for term in row.get("penalty_terms", []) if isinstance(row, dict) else []:
            if str(term).lower() in text_l:
                score -= float(row.get("penalty", default_penalty))
    return score
