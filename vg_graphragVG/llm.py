from __future__ import annotations

from typing import Protocol


class StructuredLLM(Protocol):
    def complete_json(self, prompt: str) -> dict: ...


class NoopLLM:
    def complete_json(self, prompt: str) -> dict:
        return {}
