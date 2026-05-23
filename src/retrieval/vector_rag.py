"""VectorRAG retrieval wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.indexing.vector_index import VectorIndex


class VectorRAGRetriever:
    """Vector-style retriever."""

    def __init__(self, index_dir: Path):
        self.index = VectorIndex.load(index_dir)

    def retrieve(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k vector chunks for a question."""
        return self.index.search(question, top_k=top_k)
