from __future__ import annotations

import re
from typing import Dict, List

from vg_graphrag.models import Edge, EvidencePackage, EvidencePath, Node, QueryAnalysis, TextChunk, ToolResult
from vg_graphrag.runtime_skill import pattern_term_score


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "that", "this", "what", "which", "how", "why", "does", "do", "did", "are", "is"}
    return {t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "") if len(t) > 2 and t.lower() not in stop}


def _analysis_terms(analysis: QueryAnalysis) -> set[str]:
    terms = set()
    for ent in analysis.entities:
        terms |= _tokens(ent.text)
    for term in analysis.constraints.get("canonical_terms", []) if analysis.constraints else []:
        terms |= _tokens(str(term))
    for term in analysis.constraints.get("alias_terms", []) if analysis.constraints else []:
        terms |= _tokens(str(term))
    terms |= _tokens(analysis.answer_slot.replace("_", " "))
    return terms


def _path_relevance(path: EvidencePath, analysis: QueryAnalysis) -> float:
    terms = _analysis_terms(analysis)
    node_terms = _tokens(" ".join(path.nodes))
    edge_terms = _tokens(" ".join((edge.get("relation") or "") for edge in path.edges))
    overlap = len(terms & (node_terms | edge_terms))
    score = float(overlap)
    if any(str(n).startswith("claim::") for n in path.nodes):
        score -= 0.5
    if path.text_support_ids:
        score += 0.5
    return score


def _chunk_relevance(chunk: TextChunk, analysis: QueryAnalysis) -> float:
    disable_mechanism = bool((analysis.constraints or {}).get("disable_mechanism_ranking"))
    terms = _analysis_terms(analysis)
    text_terms = _tokens(chunk.text)
    score = float(len(terms & text_terms))
    skill = dict(((analysis.constraints or {}).get("runtime_skill") or {}))
    matched_patterns = set((((analysis.constraints or {}).get("domain_hints") or {}).get("matched_patterns")) or [])
    source_type = str(chunk.provenance.get("source_type", "")).lower()
    if source_type == "direct_qa_train":
        score += 1.0
    text_l = chunk.text.lower()
    score += pattern_term_score(skill, matched_patterns, text_l, default_boost=3.5, default_penalty=2.5)
    if "question:" in text_l and "answer:" in text_l:
        score += 0.5
    text_l = chunk.text.lower()
    if "question:" in text_l and "answer:" in text_l:
        score -= 1.0
    if not disable_mechanism and analysis.answer_slot in {"reproduction", "mechanism", "cause", "connection"}:
        if any(x in text_l for x in ["progesterone", "corpus luteum", "embryo", "embryonic", "endometrium", "uterine", "metabolic", "endocrine", "ovulation"]):
            score += 5.0
    if any(x in text_l for x in ["feed intake", "feeding environment", "feed bunk", "competition for feed", "social stress", "housing"]):
        score += 2.0
    if "feed costs usually account" in text_l:
        score -= 2.0
    if not disable_mechanism and analysis.answer_slot in {"management", "economic"}:
        if any(x in text_l for x in ["planned production", "auction market prices", "predict operating profits", "supply and demand", "revenue", "cash flow", "market readiness", "market alignment", "workflow", "coordination"]):
            score += 5.0
    if not disable_mechanism and analysis.answer_slot in {"nutrition", "management"}:
        if any(x in text_l for x in ["metabolizable energy", "digestible nutrients", "digestibility", "nutrient density", "nutrient synchrony", "protein utilization", "mineral ratios", "absorption", "balance"]):
            score += 5.0
    if {"goat", "kid", "doe"} & terms and {"sheep", "ewe", "lamb"} & text_terms and not ({"goat", "kid", "doe"} & text_terms):
        score -= 2.0
    return score


def _chunk_from_dict(d: dict) -> TextChunk:
    if "chunk" in d and isinstance(d["chunk"], dict):
        d = d["chunk"]
    txt = d.get("text", "") or ""
    # Prefer answer body for DirectQA-derived chunks to avoid copying unrelated question text.
    lower_txt = txt.lower()
    if "answer:" in lower_txt:
        pos = lower_txt.find("answer:")
        txt = txt[pos + len("answer:") :].strip()
    txt = txt.replace("Question:", " ").replace("Answer:", " ").strip()
    return TextChunk(d.get("chunk_id") or d.get("evidence_id") or "unknown", txt, d.get("source_document_id"), d.get("provenance", {}))


def _infer_sentence_relation(sentence: str) -> str:
    s = sentence.lower()
    mapping = [
        ("secretes", "secretes"),
        ("promotes", "promotes"),
        ("maintains", "maintains"),
        ("determines", "determines"),
        ("directly affects", "directly_affects"),
        ("affects", "affects"),
        ("influences", "influences"),
        ("causes", "causes"),
        ("results in", "results_in"),
        ("leads to", "leads_to"),
        ("reduces risk", "reduces_risk_from"),
        ("reduces risks", "reduces_risk_from"),
        ("stabilizing", "stabilizes"),
        ("stabilize", "stabilizes"),
        ("improves", "improves"),
        ("requires", "requires"),
        ("supports", "supports"),
        ("predict", "improves_predictability"),
    ]
    for needle, relation in mapping:
        if needle in s:
            return relation
    return "chunk_claim"


def _extract_chunk_claims(chunks: list[TextChunk], analysis: QueryAnalysis, limit: int = 10) -> list[dict]:
    disable_mechanism = bool((analysis.constraints or {}).get("disable_mechanism_ranking"))
    focus_terms = _analysis_terms(analysis)
    skill = dict(((analysis.constraints or {}).get("runtime_skill") or {}))
    matched_patterns = set((((analysis.constraints or {}).get("domain_hints") or {}).get("matched_patterns")) or [])
    out: list[tuple[float, dict]] = []
    for chunk in chunks:
        text = " ".join((chunk.text or "").split())
        if not text:
            continue
        source_type = str(chunk.provenance.get("source_type", "")).lower()
        for idx, sent in enumerate(re.split(r"(?<=[.!?])\s+", text)):
            s = sent.strip()
            if len(s) < 24:
                continue
            stok = _tokens(s)
            overlap = len(focus_terms & stok)
            if overlap == 0:
                continue
            score = float(overlap)
            sl = s.lower()
            score += pattern_term_score(skill, matched_patterns, sl, default_boost=3.5, default_penalty=2.5)
            if (not disable_mechanism) and analysis.answer_slot in {"reproduction", "mechanism", "cause"} and any(x in sl for x in ["progesterone", "corpus luteum", "embryo", "embryonic", "uterine", "ovulation"]):
                score += 4.0
            if (not disable_mechanism) and analysis.answer_slot in {"management", "economic"} and any(x in sl for x in ["planned production", "market", "auction", "cash flow", "profits and losses", "supply and demand", "predict operating profits", "market prices"]):
                score += 4.0
            if (not disable_mechanism) and analysis.answer_slot in {"nutrition", "management"} and any(x in sl for x in ["feed intake", "feed bunk", "competition", "feeding environment", "social stress", "housing"]):
                score += 3.0
            if source_type == "corpus_doc":
                score += 0.75
            elif source_type == "direct_qa_train":
                score += 0.25
            claim = {
                "claim_id": f"chunkclaim::{chunk.chunk_id}::{idx}",
                "claim_text": s,
                "head": "",
                "relation": _infer_sentence_relation(s),
                "tail": "",
                "supporting_quote": s,
                "source_chunk_id": chunk.chunk_id,
                "source_document_id": chunk.source_document_id,
                "source_type": chunk.provenance.get("source_type"),
                "source_question_id": chunk.provenance.get("source_question_id"),
                "confidence": None,
                "match_score": score,
                "provenance": dict(chunk.provenance or {}),
            }
            out.append((score, claim))
    out.sort(key=lambda x: (-x[0], str(x[1].get("claim_text") or "")))
    deduped: list[dict] = []
    seen = set()
    for _, claim in out:
        sig = str(claim.get("claim_text") or "").lower()
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(claim)
        if len(deduped) >= limit:
            break
    return deduped


def _edge_from_dict(d: dict) -> Edge:
    return Edge(
        d.get("edge_id") or d.get("relation_id") or f"{d.get('source')}->{d.get('target')}",
        d.get("source") or d.get("head") or d.get("source_id") or "",
        d.get("target") or d.get("tail") or d.get("target_id") or "",
        d.get("relation") or d.get("type") or "",
        d.get("supporting_quote") or "",
        d.get("source_chunk_id"),
        d.get("source_document_id"),
        d.get("confidence"),
        d.get("provenance", {}),
    )


def build_evidence_package(tool_history: List[ToolResult], analysis: QueryAnalysis) -> EvidencePackage:
    nodes: Dict[str, Node] = {}
    edges: Dict[str, Edge] = {}
    paths: List[EvidencePath] = []
    chunks: Dict[str, TextChunk] = {}
    supporting_claims: Dict[str, dict] = {}
    entity_found = False

    for tr in tool_history:
        if tr.tool_name == "EntitySearch" and tr.results:
            entity_found = True
            for r in tr.results:
                n = r.get("node")
                if isinstance(n, Node):
                    nodes[n.node_id] = n
        elif tr.tool_name == "GraphNeighbor":
            for group in tr.results:
                for nd in group.get("nodes", []):
                    nodes[nd["node_id"]] = Node(nd["node_id"], nd.get("name", nd["node_id"]), nd.get("aliases", []), nd.get("node_type", "entity"), nd.get("provenance", {}))
                for ed in group.get("edges", []):
                    e = _edge_from_dict(ed)
                    edges[e.edge_id] = e
        elif tr.tool_name == "PathSearch":
            for p in tr.results:
                es = [_edge_from_dict(e) for e in p.get("edges", [])]
                for e in es:
                    edges[e.edge_id] = e
                paths.append(EvidencePath(p.get("nodes", []), [e.__dict__ for e in es], [e.source_chunk_id for e in es if e.source_chunk_id]))
        elif tr.tool_name == "ClaimSearch":
            for claim in tr.results:
                cid = str(claim.get("claim_id") or "")
                if not cid:
                    continue
                claim_copy = dict(claim)
                claim_copy["_subquery_id"] = tr.query.get("subquery_id")
                supporting_claims[cid] = claim_copy
                head = str(claim.get("head") or "")
                tail = str(claim.get("tail") or "")
                if head and head not in nodes:
                    nodes[head] = Node(head, head, [], "entity", claim.get("provenance", {}))
                if tail and tail not in nodes:
                    nodes[tail] = Node(tail, tail, [], "entity", claim.get("provenance", {}))
                if head and tail:
                    edge = Edge(
                        cid,
                        head,
                        tail,
                        str(claim.get("relation") or "related_to"),
                        str(claim.get("supporting_quote") or ""),
                        claim.get("source_chunk_id"),
                        claim.get("source_document_id"),
                        claim.get("confidence"),
                        claim.get("provenance", {}),
                    )
                    edges[edge.edge_id] = edge
        elif tr.tool_name in {"TextSearch", "HybridSearch"}:
            for r in tr.results:
                c = _chunk_from_dict(r)
                c.provenance = dict(c.provenance or {})
                c.provenance["tool_query_subquery_id"] = tr.query.get("subquery_id")
                chunks[c.chunk_id] = c
                linked = r.get("linked_edge") if isinstance(r, dict) else None
                if linked:
                    e = _edge_from_dict(linked)
                    edges[e.edge_id] = e

    for e in edges.values():
        if e.source_chunk_id and e.source_chunk_id not in chunks:
            # Retain provenance signal even if text body is unavailable.
            chunks[e.source_chunk_id] = TextChunk(e.source_chunk_id, e.supporting_quote or "", e.source_document_id, e.provenance)

    if paths:
        ranked_paths = sorted((( _path_relevance(p, analysis), p) for p in paths), key=lambda x: x[0], reverse=True)
        if analysis.constraints.get("canonical_terms"):
            ranked_paths = [x for x in ranked_paths if x[0] > 0] or ranked_paths
        paths = [p for _, p in ranked_paths[:8]]
    if chunks:
        ranked_chunks = sorted(chunks.values(), key=lambda c: _chunk_relevance(c, analysis), reverse=True)
        ranked_chunks = [c for c in ranked_chunks if _chunk_relevance(c, analysis) > -1.0]
        chunks = {c.chunk_id: c for c in ranked_chunks[:20]}
        for claim in _extract_chunk_claims(list(chunks.values()), analysis, limit=12):
            supporting_claims.setdefault(str(claim.get("claim_id") or ""), claim)

    path_found = bool(paths) or bool(edges)
    text_support = any(c.text.strip() for c in chunks.values()) or any(str(c.get("supporting_quote") or "").strip() for c in supporting_claims.values())
    weak_prov = any(not (e.source_chunk_id or e.supporting_quote) for e in edges.values()) if edges else False
    entity_coverage = 1.0 if entity_found or nodes else 0.0
    path_completeness = 1.0 if paths else 0.75 if supporting_claims else 0.5 if edges else 0.0
    text_support_score = 1.0 if text_support else 0.0
    if edges:
        rel_rel = 1.0 if any(analysis.answer_slot in (e.relation or "").lower() for e in edges.values()) else 0.5
    else:
        rel_rel = 0.0
    if supporting_claims and any((c.get("source_type") == "direct_qa_train") for c in supporting_claims.values()):
        source_rel = 1.0
    else:
        source_rel = 0.5 if weak_prov else 1.0 if (chunks or edges or supporting_claims) else 0.0
    components = {
        "entity_coverage": entity_coverage,
        "path_completeness": path_completeness,
        "text_support": text_support_score,
        "relation_relevance": rel_rel,
        "source_reliability": source_rel,
        "recency": 0.5,
    }
    score = (
        0.25 * components["entity_coverage"]
        + 0.25 * components["path_completeness"]
        + 0.20 * components["text_support"]
        + 0.15 * components["relation_relevance"]
        + 0.10 * components["source_reliability"]
        + 0.05 * components["recency"]
    )
    claims = list(supporting_claims.values()) or [{"text": c.text[:240], "source_chunk_id": c.chunk_id} for c in chunks.values() if c.text]
    return EvidencePackage(
        claim_candidates=claims,
        supporting_claims=list(supporting_claims.values()),
        supporting_paths=paths,
        supporting_chunks=list(chunks.values()),
        subgraph_nodes=list(nodes.values()),
        subgraph_edges=list(edges.values()),
        coverage_flags={
            "source_entity_found": bool(nodes) or entity_found,
            "target_entity_found": len(nodes) > 1,
            "path_found": path_found,
            "text_support_found": text_support,
        },
        evidence_score=score,
        score_components=components,
        noise_flags={"weak_provenance": weak_prov, "excessive_noise": False},
        provenance_summary={
            "chunk_count": len(chunks),
            "edge_count": len(edges),
            "path_count": len(paths),
            "claim_count": len(supporting_claims),
            "weak_provenance": weak_prov,
        },
    )
