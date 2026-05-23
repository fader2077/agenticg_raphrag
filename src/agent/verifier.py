"""Evidence verifier for deterministic AgenticGraphRAG."""

from __future__ import annotations

import re
from typing import Any


def verify_answer(question: str, answer: str, evidence_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether an answer is text-supported by retrieved evidence."""
    if not evidence_chunks:
        return {
            "verdict": "insufficient_evidence",
            "unsupported_claims": [answer] if answer else [],
            "required_missing_evidence": ["retrieved evidence"],
            "trace": {"evidence_chunk_count": 0, "answer_token_coverage": 0.0},
        }
    if not answer or answer == "insufficient evidence":
        return {
            "verdict": "insufficient_evidence",
            "unsupported_claims": [],
            "required_missing_evidence": ["answer"],
            "trace": {"evidence_chunk_count": len(evidence_chunks), "answer_token_coverage": 0.0},
        }
    joined = " ".join(chunk.get("text", "") for chunk in evidence_chunks).lower()
    answer_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", answer) if len(t) > 2]
    if not answer_tokens:
        return {
            "verdict": "partially_supported",
            "unsupported_claims": [answer],
            "required_missing_evidence": [],
            "trace": {"evidence_chunk_count": len(evidence_chunks), "answer_token_coverage": 0.0},
        }
    coverage = sum(1 for token in answer_tokens if token in joined) / len(answer_tokens)
    if coverage >= 0.8 or answer.lower() in {"yes", "no"}:
        verdict = "supported"
    elif coverage > 0:
        verdict = "partially_supported"
    else:
        verdict = "unsupported"
    return {
        "verdict": verdict,
        "unsupported_claims": [] if verdict == "supported" else [answer],
        "required_missing_evidence": [],
        "trace": {"evidence_chunk_count": len(evidence_chunks), "answer_token_coverage": coverage},
    }
