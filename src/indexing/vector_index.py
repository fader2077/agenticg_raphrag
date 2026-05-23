"""Dense/vector-style indexing with Ollama or sentence-transformers support."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np


class VectorIndex:
    """A persisted vector index for HotpotQA chunks."""

    def __init__(self, backend: str, chunks: list[dict[str, Any]], model: Any, matrix: Any, faiss_index: Any = None):
        self.backend = backend
        self.chunks = chunks
        self.model = model
        self.matrix = matrix
        self.faiss_index = faiss_index

    @classmethod
    def build(
        cls,
        chunks: list[dict[str, Any]],
        backend: str = "tfidf_char",
        embed_model: str | None = None,
        ollama_host: str | None = None,
        fail_on_error: bool = False,
    ) -> "VectorIndex":
        """Build a vector index, supporting true dense embeddings when requested."""
        texts = [chunk.get("text", "") for chunk in chunks]
        dense_backend = backend in {"ollama_embeddings", "ollama_dense", "dense_ollama"}
        if dense_backend:
            try:
                from ollama import Client

                model_name = embed_model or os.environ.get("HOTPOTQA_OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
                client = Client(host=ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
                matrix = np.asarray([client.embeddings(model=model_name, prompt=text or " ")["embedding"] for text in texts], dtype="float32")
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.clip(norms, 1e-12, None)
                return cls("ollama_embeddings", chunks, {"model_name": model_name, "host": ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")}, matrix, None)
            except Exception:
                if fail_on_error:
                    raise
        use_st = backend == "sentence_transformers" or os.environ.get("HOTPOTQA_USE_SENTENCE_TRANSFORMERS") == "1"
        if use_st:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.environ.get("HOTPOTQA_SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                model = SentenceTransformer(model_name)
                matrix = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
                return cls("sentence_transformers", chunks, {"model_name": model_name, "model": model}, matrix, None)
            except Exception:
                if fail_on_error:
                    raise

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2")
        matrix = vectorizer.fit_transform(texts)
        return cls("tfidf_char_vector", chunks, vectorizer, matrix, None)

    def save(self, path: Path) -> None:
        """Save the vector index."""
        path.mkdir(parents=True, exist_ok=True)
        payload = {"backend": self.backend, "chunks": self.chunks, "model": self.model, "matrix": self.matrix}
        if self.backend in {"sentence_transformers", "ollama_embeddings"}:
            payload["model"] = {"model_name": self.model["model_name"]}
            if self.backend == "ollama_embeddings":
                payload["model"]["host"] = self.model.get("host")
        with (path / "vector_index.pkl").open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        """Load a vector index."""
        with (path / "vector_index.pkl").open("rb") as f:
            payload = pickle.load(f)
        return cls(payload["backend"], payload["chunks"], payload["model"], payload["matrix"], None)

    def _query_vector(self, query: str) -> Any:
        if self.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            model_name = self.model["model_name"] if isinstance(self.model, dict) else self.model.get("model_name")
            model = SentenceTransformer(model_name)
            return model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        if self.backend == "ollama_embeddings":
            from ollama import Client

            model_name = self.model["model_name"] if isinstance(self.model, dict) else self.model.get("model_name")
            host = self.model.get("host") if isinstance(self.model, dict) else None
            client = Client(host=host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
            vec = np.asarray(client.embeddings(model=model_name, prompt=query or " ")["embedding"], dtype="float32")
            vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
            return vec.reshape(1, -1)
        return self.model.transform([query])

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search chunks and return ranked vector retrieval records."""
        if not self.chunks:
            return []
        q = self._query_vector(query)
        if self.backend in {"sentence_transformers", "ollama_embeddings"}:
            scores = np.asarray(self.matrix @ q[0], dtype=float)
        else:
            scores = (self.matrix @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:top_k]
        out: list[dict[str, Any]] = []
        for rank, idx in enumerate(order, start=1):
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = float(scores[int(idx)])
            chunk["rank"] = rank
            chunk["source"] = "vector_search"
            out.append(chunk)
        return out
