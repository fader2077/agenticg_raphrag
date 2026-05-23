"""AgenticGraphRAG aggregate metrics."""

from __future__ import annotations

from typing import Any


def agent_case_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Compute per-case agent tool and verifier metrics."""
    trace = record.get("tool_trace") or []
    verifier = record.get("verifier") or {}
    repair_used = any(t.get("tool") == "repair_retrieve" for t in trace)
    final_supported = verifier.get("verdict") == "supported" and record.get("pred_answer") != "insufficient evidence"
    return {
        "tool_calls": float(len(trace)),
        "repair_used": 1.0 if repair_used else 0.0,
        "repair_success": 1.0 if repair_used and final_supported else 0.0,
        "verifier_unsupported": 1.0 if verifier.get("verdict") in {"unsupported", "partially_supported", "insufficient_evidence"} else 0.0,
        "final_unsupported": 1.0 if record.get("pred_answer") == "insufficient evidence" else 0.0,
    }
