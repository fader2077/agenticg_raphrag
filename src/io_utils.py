"""Shared IO and configuration helpers for reproducible HotpotQA runs."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


METHODS = [
    "LLM-only",
    "TextRAG",
    "VectorRAG",
    "GraphRAG-hop1",
    "GraphRAG-hop2",
    "AgenticGraphRAG",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records from a path."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write records as UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON with a caller-supplied default for missing files."""
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write a pretty JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write YAML config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def suffix_name(name: str, suffix: str | None) -> str:
    """Append a sample suffix to a path component without duplicating it."""
    if not suffix:
        return name
    return name if name.endswith(f"_{suffix}") else f"{name}_{suffix}"


def processed_dir(config: dict[str, Any], sample_suffix: str | None = None) -> Path:
    """Resolve the processed HotpotQA directory for a suffix."""
    base = Path(config["data"]["processed_dir"])
    if not sample_suffix:
        return base
    return base.parent / suffix_name(base.name, sample_suffix)


def index_dir(config: dict[str, Any], key: str, sample_suffix: str | None = None) -> Path:
    """Resolve an index directory, mapping hotpotqa -> hotpotqa_suffix when requested."""
    base = Path(config["indexing"][key])
    if not sample_suffix:
        return base
    parts = list(base.parts)
    for i, part in enumerate(parts):
        if part == "hotpotqa":
            parts[i] = suffix_name(part, sample_suffix)
            return Path(*parts)
    return base.parent / suffix_name(base.name, sample_suffix)


def run_dir(config: dict[str, Any], sample_suffix: str | None = None) -> Path:
    """Resolve run directory for a config and optional suffix."""
    name = config.get("experiment", {}).get("name", "hotpotqa_round1")
    return Path("runs") / suffix_name(name, sample_suffix)


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def run_capture(cmd: list[str], cwd: Path | None = None) -> dict[str, Any]:
    """Run a subprocess and return captured output without raising."""
    started = os.times()
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, encoding="utf-8", errors="replace")
        elapsed = max(0.0, os.times().elapsed - started.elapsed)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": 999, "stdout": "", "stderr": str(exc), "elapsed_sec": 0.0}


def collect_repo_inspection() -> dict[str, Any]:
    """Collect repository and environment inspection artifacts for manifests."""
    commands = {
        "git_status": ["git", "status", "--short"],
        "git_branch": ["git", "branch", "--show-current"],
        "git_ls_files": ["git", "ls-files"],
        "python_version": ["python", "--version"],
        "pip_freeze": ["python", "-m", "pip", "freeze"],
    }
    out = {name: run_capture(cmd) for name, cmd in commands.items()}
    required = [
        "judge_binary_correctness.py",
        "judge_openai.py",
        "config.py",
        "builder.py",
        "src",
        "scripts",
        "eval",
        "data",
        "runs",
        "configs",
    ]
    out["required_path_exists"] = {path: Path(path).exists() for path in required}
    return out
