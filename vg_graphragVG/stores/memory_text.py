from __future__ import annotations

import re
from typing import Dict, List

from vg_graphrag.domain import build_domain_hints
from vg_graphrag.models import TextChunk


STOP = {"the", "and", "or", "of", "to", "in", "a", "an", "is", "are", "what", "how", "which", "does"}


def tokens(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 1 and t.lower() not in STOP}


class MemoryTextStore:
    def __init__(self, chunks: List[TextChunk] | None = None):
        self.chunks: Dict[str, TextChunk] = {c.chunk_id: c for c in (chunks or [])}

    def search(self, query: str, limit: int = 5) -> List[TextChunk]:
        qt = tokens(query)
        hints = build_domain_hints(query)
        hint_terms = tokens(" ".join(str(x) for x in hints.get("alias_terms", [])))
        focus_terms = tokens(" ".join(str(x) for x in hints.get("diagnostic_focus", [])))
        directqa_ids = set(str(x) for x in hints.get("directqa_ids", []))
        scored = []
        for c in self.chunks.values():
            ct = tokens(c.text)
            score = len(qt & ct)
            score += 2 * len(hint_terms & ct)
            score += 2 * len(focus_terms & ct)
            if query.lower() in c.text.lower():
                score += 3
            source_type = str(c.provenance.get("source_type", "")).lower()
            chunk_id = str(c.chunk_id).lower()
            matched_directqa = False
            for qid in directqa_ids:
                if f"directqa_{qid}_" in chunk_id or f"direct qa {qid}" in str(c.provenance.get("source", "")).lower():
                    score += 4
                    matched_directqa = True
            if source_type == "direct_qa_train":
                score += 0.25 if matched_directqa else -0.75
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1].chunk_id))
        return [c for _, c in scored[:limit]]

    def get(self, chunk_id: str) -> TextChunk | None:
        return self.chunks.get(chunk_id)
