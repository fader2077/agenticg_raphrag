"""TextRAG retrieval wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.indexing.text_index import TextIndex


class TextRAGRetriever:
    """BM25/TF-IDF text retriever."""

    def __init__(self, index_dir: Path):
        self.index = TextIndex.load(index_dir)

    def retrieve(self, question: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve top-k text chunks for a question."""
        return self.index.search(question, top_k=top_k)
