from __future__ import annotations

from vg_graphrag.models import EvidencePackage, QueryAnalysis


def synthesize_patch(question: str, base_answer: str, evidence: EvidencePackage, analysis: QueryAnalysis) -> str:
    if not evidence.supporting_chunks:
        return ""
    preferred_ids = []
    for p in evidence.supporting_paths:
        preferred_ids.extend(p.text_support_ids)
    preferred = [c for c in evidence.supporting_chunks if c.chunk_id in preferred_ids]
    candidates = preferred or evidence.supporting_chunks
    chosen = ""
    for c in candidates:
        txt = " ".join((c.text or "").split())
        if txt and txt.lower() not in (base_answer or "").lower():
            chosen = txt
            break
    if not chosen:
        return ""
    sentence = chosen.split(".")[0].strip()
    if not sentence:
        return ""
    words = sentence.split()
    if len(words) > 45:
        sentence = " ".join(words[:45])
    return f"Additional verified evidence indicates that {sentence}."
