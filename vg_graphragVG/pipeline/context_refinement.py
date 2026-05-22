from __future__ import annotations

import re

from vg_graphrag.models import QueryAnalysis
from vg_graphrag.models import ChannelEvidence, ToolResult


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "what", "which", "how", "goat", "goats"}
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 2 and t.lower() not in stop}


def _text_of(item: dict) -> str:
    if "chunk" in item and isinstance(item["chunk"], dict):
        return item["chunk"].get("text", "") or ""
    if "claim_text" in item:
        return item.get("supporting_quote", "") or item.get("claim_text", "") or ""
    return item.get("text", "") or item.get("supporting_quote", "") or ""


def _score_text(query: str, item: dict, analysis: QueryAnalysis | None = None) -> float:
    q = _tokens(query)
    constraints = (analysis.constraints or {}) if analysis else {}
    hints = dict(constraints.get("domain_hints") or {})
    hint_tokens = _tokens(" ".join(str(x) for x in hints.get("alias_terms", [])))
    text = _text_of(item)
    if not q or not text:
        return 0.0
    t = _tokens(text)
    score = float(len(q & t)) / max(1.0, len(q))
    if hint_tokens:
        score += 1.5 * float(len(hint_tokens & t)) / max(1.0, len(hint_tokens))
    ql = query.lower()
    tl = text.lower()
    if any(x in ql for x in ["goat", "goats", "doe", "does", "kid", "kids"]):
        if any(x in tl for x in ["sheep", "ewe", "ewes", "lamb", "lambs"]) and not any(x in tl for x in ["goat", "goats", "doe", "does", "kid", "kids"]):
            score -= 0.5
    if constraints.get("enable_directqa_linking") and hints.get("directqa_ids"):
        chunk_id_l = str(item.get("chunk_id", "")).lower()
        prov_l = str(item.get("provenance", {})).lower()
        for qid in hints.get("directqa_ids", []):
            if f"directqa_{qid}_" in chunk_id_l or f"direct qa {qid}" in prov_l:
                score += 1.0
                break
    return score


def _score_relational(query: str, item: dict, analysis: QueryAnalysis | None = None) -> float:
    score = 0.0
    if item.get("claim"):
        claim = item.get("claim") or {}
        text = " ".join(
            [
                str(claim.get("claim_text") or ""),
                str(claim.get("head") or ""),
                str(claim.get("relation") or ""),
                str(claim.get("tail") or ""),
                str(claim.get("supporting_quote") or ""),
            ]
        )
        score += 6.0 + _score_text(query, {"text": text}, analysis)
        if claim.get("supporting_quote"):
            score += 1.0
        if claim.get("source_type") == "direct_qa_train":
            score += 0.5
        return score
    if item.get("edges"):
        text = " ".join(
            " ".join([str(e.get("source", "")), str(e.get("relation", "")), str(e.get("target", "")), str(e.get("supporting_quote", ""))])
            for e in item.get("edges", [])
        )
        score += 3.0 + _score_text(query, {"text": text}, analysis)
        return score
    if item.get("name"):
        text = str(item.get("name") or "")
        score += 1.0 + _score_text(query, {"text": text}, analysis)
    return score


def refine_context(subquery_id: str, grounded_query: str, results: list[ToolResult], max_items: int = 5, analysis: QueryAnalysis | None = None) -> ChannelEvidence:
    """CR: keep focused semantic and relational evidence for one subquery."""
    semantic: list[dict] = []
    relational: list[dict] = []
    for tr in results:
        if tr.tool_name in {"TextSearch", "HybridSearch"}:
            semantic.extend(dict(x) for x in tr.results)
            for x in tr.results:
                linked = x.get("linked_edge") if isinstance(x, dict) else None
                if linked:
                    relational.append({"edges": [linked], "nodes": [linked.get("source"), linked.get("target")]})
        elif tr.tool_name == "PathSearch":
            relational.extend(dict(x) for x in tr.results)
        elif tr.tool_name == "GraphNeighbor":
            for group in tr.results:
                relational.append({"node_id": group.get("node_id"), "nodes": group.get("nodes", []), "edges": group.get("edges", [])})
        elif tr.tool_name == "ClaimSearch":
            for claim in tr.results[: max_items * 2]:
                pseudo_chunk = {
                    "chunk_id": claim.get("source_chunk_id") or claim.get("claim_id"),
                    "text": claim.get("supporting_quote") or claim.get("claim_text") or "",
                    "source_document_id": claim.get("source_document_id"),
                    "provenance": {
                        **(claim.get("provenance") or {}),
                        "source_tool": "ClaimSearch",
                        "claim_id": claim.get("claim_id"),
                    },
                }
                semantic.append({"chunk": pseudo_chunk, "claim": claim})
                relational.append(
                    {
                        "claim": claim,
                        "nodes": [claim.get("head"), claim.get("tail")],
                        "edges": [
                            {
                                "edge_id": claim.get("claim_id"),
                                "source": claim.get("head"),
                                "relation": claim.get("relation"),
                                "target": claim.get("tail"),
                                "supporting_quote": claim.get("supporting_quote"),
                                "source_chunk_id": claim.get("source_chunk_id"),
                                "source_document_id": claim.get("source_document_id"),
                                "confidence": claim.get("confidence"),
                                "provenance": claim.get("provenance", {}),
                            }
                        ],
                    }
                )
        elif tr.tool_name == "EntitySearch":
            relational.extend(dict(x) for x in tr.results[:max_items])

    semantic.sort(key=lambda x: _score_text(grounded_query, x, analysis), reverse=True)
    semantic = semantic[:max_items]
    relational.sort(key=lambda x: _score_relational(grounded_query, x, analysis), reverse=True)
    relational = relational[:max_items]
    semantic_summary = " ".join(_text_of(x).strip().replace("\n", " ")[:220] for x in semantic if _text_of(x))[:900]
    rel_bits = []
    for item in relational:
        if item.get("edges"):
            for e in item.get("edges", [])[:3]:
                rel_bits.append(f"{e.get('source')} -{e.get('relation') or e.get('type')}-> {e.get('target')}")
        elif item.get("name"):
            rel_bits.append(str(item.get("name")))
    relational_summary = "; ".join(x for x in rel_bits if x)[:900]
    missing = []
    if not semantic:
        missing.append("semantic_text_evidence")
    if not relational:
        missing.append("relational_graph_evidence")
    return ChannelEvidence(subquery_id, grounded_query, semantic, relational, semantic_summary, relational_summary, missing)
