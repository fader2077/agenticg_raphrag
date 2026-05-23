"""Tool wrappers with trace capture for AgenticGraphRAG."""

from __future__ import annotations

import time
from typing import Any, Callable


def call_tool(trace: list[dict[str, Any]], name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a tool and append a status/latency trace record."""
    start = time.perf_counter()
    record = {"tool": name, "input": kwargs.get("trace_input", args[0] if args else ""), "status": "ok"}
    try:
        result = fn(*args, **{k: v for k, v in kwargs.items() if k != "trace_input"})
        if isinstance(result, list):
            record["num_results"] = len(result)
        elif isinstance(result, dict):
            record["num_results"] = len(result.get("retrieved_paths") or result.get("graph_evidence_chunks") or result)
        return result
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["latency_ms"] = int((time.perf_counter() - start) * 1000)
        trace.append(record)
