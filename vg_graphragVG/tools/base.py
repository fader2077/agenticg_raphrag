from __future__ import annotations

from typing import Protocol

from vg_graphrag.models import AgentState, ToolResult


class RetrieverTool(Protocol):
    name: str

    def run(self, input: dict, state: AgentState) -> ToolResult: ...
