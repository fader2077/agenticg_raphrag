from __future__ import annotations

from typing import List, Protocol

from vg_graphrag.models import TextChunk


class TextStore(Protocol):
    def search(self, query: str, limit: int = 5) -> List[TextChunk]: ...

    def get(self, chunk_id: str) -> TextChunk | None: ...
