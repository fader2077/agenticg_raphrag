from __future__ import annotations

from vg_graphrag.models import AgentState, ToolResult


class StubTool:
    def __init__(self, name: str):
        self.name = name

    def run(self, input: dict, state: AgentState) -> ToolResult:
        return ToolResult(self.name, input, results=[], errors=[f"{self.name} is a v1 stub"])
