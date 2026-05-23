"""BM25 text indexing with sklearn TF-IDF fallback."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25-style retrieval."""
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


class TextIndex:
    """A persisted sparse text index for HotpotQA chunks."""

    def __init__(self, backend: str, chunks: list[dict[str, Any]], model: Any, matrix: Any = None):
        self.backend = backend
        self.chunks = chunks
        self.model = model
        self.matrix = matrix

    @classmethod
    def build(cls, chunks: list[dict[str, Any]]) -> "TextIndex":
        """Build a BM25 index when available, otherwise a word TF-IDF index."""
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [tokenize(chunk.get("text", "")) for chunk in chunks]
            return cls("rank_bm25", chunks, BM25Okapi(tokenized), None)
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = vectorizer.fit_transform([chunk.get("text", "") for chunk in chunks])
            return cls("sklearn_tfidf", chunks, vectorizer, matrix)

    def save(self, path: Path) -> None:
        """Save the text index."""
        path.mkdir(parents=True, exist_ok=True)
        with (path / "text_index.pkl").open("wb") as f:
            pickle.dump({"backend": self.backend, "chunks": self.chunks, "model": self.model, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path: Path) -> "TextIndex":
        """Load a text index."""
        with (path / "text_index.pkl").open("rb") as f:
            payload = pickle.load(f)
        return cls(payload["backend"], payload["chunks"], payload["model"], payload.get("matrix"))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search chunks and return ranked retrieval records."""
        if not self.chunks:
            return []
        if self.backend == "rank_bm25":
            scores = np.asarray(self.model.get_scores(tokenize(query)), dtype=float)
        else:
            q = self.model.transform([query])
            scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:top_k]
        out: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            chunk["rank"] = rank
            chunk["source"] = "text_search"
            out.append(chunk)
        return out
