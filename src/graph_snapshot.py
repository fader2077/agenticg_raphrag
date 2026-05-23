"""Minimal graph snapshot compatibility for legacy builder.py imports."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return sha256 for a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class GraphSnapshotManager:
    """Compatibility no-op snapshot manager."""

    def __init__(self, neo4j_database: str = "neo4j") -> None:
        self.neo4j_database = neo4j_database

    def create_run_context(self, **kwargs: Any) -> Any:
        """Return a simple object-like run context."""
        return type("RunContext", (), kwargs)()

    def try_neo4j_dump(self, run_ctx: Any) -> dict[str, Any]:
        """Return an unavailable dump status without failing legacy callers."""
        return {"status": "skipped", "reason": "local HotpotQA pipeline uses NetworkX fallback", "path": None}
