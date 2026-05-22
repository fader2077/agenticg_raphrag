import hashlib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from ollama import Client

from config import CONFIG, TRIPLE_PROMPT_TEMPLATE
from src.database import ensure_entity_index, ensure_fulltext_index, ensure_graph_run_index, ensure_vector_index
from src.graph_snapshot import GraphSnapshotManager, sha256_file
from src.models import OllamaVectorEmbedder
from src.utils import chunk_text, normalize_text, strip_think_tokens


DEFAULT_CHUNK_SIZE = CONFIG["optimal_indexing"]["chunk_size"]
DEFAULT_CHUNK_OVERLAP = CONFIG["optimal_indexing"]["overlap"]
DATASET_ID = CONFIG["infrastructure"]["dataset_id"]
GENERATION_CONFIG = CONFIG.get("generation", {})
GRAPH_INDEXING_CONFIG = CONFIG.get("graph_indexing", {})
SUPER_NODE_NAME_NORMS = {"goat", "animal", "livestock", "disease", "feed", "management"}
SUPER_NODE_ALLOW_RELATIONS = {"is_breed_of", "part_of", "used_for"}
ENTITY_SINGULAR_MAP = {
    "lambs": "lamb",
    "kids": "kid",
    "goats": "goat",
    "diseases": "disease",
    "feeds": "feed",
}
RELATION_CANONICAL_MAP = {
    "cause": "causes",
    "require": "requires",
    "leads_to": "causes",
    "results_in": "causes",
    "transmits": "transmitted_by",
    "treats": "treated_with",
    "treated_by": "treated_with",
    "symptoms": "symptoms_include",
    "symptom": "symptoms_include",
}

MIN_RELATION_LEN = 3
MAX_RELATION_LEN = 48
GENERIC_RELATIONS = {
    "is",
    "has",
    "have",
    "about",
    "related",
    "related_to",
    "associated_with",
    "mentions",
    "describes",
    "refers_to",
    "involves",
    "includes",
    "contains_information_about",
}


def load_chunks(path: Path, chunk_size: int = None, overlap: int = None) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    ovlp = overlap if overlap is not None else DEFAULT_CHUNK_OVERLAP

    print(f"    Chunking strategy: size={size}, overlap={ovlp}")

    raw_text = path.read_text(encoding="utf-8")
    segments = chunk_text(raw_text, size, ovlp)

    docs: List[Dict[str, str]] = []
    for idx, segment in enumerate(segments):
        text = segment.strip()
        doc_id = f"{DATASET_ID}_chunk_{idx:05d}"
        docs.append(
            {
                "id": doc_id,
                "document_id": path.stem,
                "text": text,
                "source": path.name,
                "source_type": "corpus_doc",
                "source_question_id": None,
                "category": None,
                "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return docs


def load_chunks_from_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = None,
    overlap: int = None,
) -> List[Dict[str, str]]:
    size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
    ovlp = overlap if overlap is not None else DEFAULT_CHUNK_OVERLAP
    docs: List[Dict[str, str]] = []
    for item in documents:
        document_id = str(item.get("document_id") or item.get("id") or "")
        if not document_id:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        source = str(item.get("title") or document_id)
        metadata = item.get("metadata") or {}
        segments = chunk_text(text, size, ovlp)
        for idx, segment in enumerate(segments):
            seg_text = segment.strip()
            chunk_id = f"{document_id}_chunk_{idx:04d}"
            docs.append(
                {
                    "id": chunk_id,
                    "document_id": document_id,
                    "text": seg_text,
                    "source": source,
                    "source_type": metadata.get("source_type", "unknown"),
                    "source_question_id": metadata.get("question_id"),
                    "category": metadata.get("category"),
                    "hash": hashlib.sha256(seg_text.encode("utf-8")).hexdigest(),
                }
            )
    return docs


def upsert_chunks(
    driver,
    embedder: OllamaVectorEmbedder,
    docs: List[Dict[str, str]],
    run_id: str = "",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    changed_docs: List[Dict[str, str]] = []
    skipped_docs: List[Dict[str, str]] = []

    with driver.session() as session:
        for doc in docs:
            existing = session.run(
                "MATCH (c:Chunk {id:$id}) RETURN c.text_hash AS hash",
                id=doc["id"],
            ).single()
            if existing and existing.get("hash") == doc["hash"]:
                skipped_docs.append(doc)
                continue

            embedding = embedder.embed_query(doc["text"])
            session.run(
                """
                MERGE (c:Chunk {id:$id})
                SET c.chunk_id = $id,
                    c.text = $text,
                    c.source = $source,
                    c.dataset = $dataset,
                    c.embedding = $embedding,
                    c.text_hash = $hash,
                    c.run_id = $run_id,
                    c.graph_run_id = $run_id,
                    c.source_document_id = $source_document_id,
                    c.source_type = $source_type,
                    c.source_question_id = $source_question_id,
                    c.category = $category,
                    c.updated_at = timestamp()
                """,
                id=doc["id"],
                text=doc["text"],
                source=doc["source"],
                dataset=DATASET_ID,
                embedding=embedding,
                hash=doc["hash"],
                run_id=run_id,
                source_document_id=doc.get("document_id") or doc["id"],
                source_type=doc.get("source_type", "unknown"),
                source_question_id=doc.get("source_question_id"),
                category=doc.get("category"),
            )
            changed_docs.append(doc)

    return changed_docs, skipped_docs


def _normalize_graph_key(value: str) -> str:
    text = normalize_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _normalize_entity_text(value: str, chunk_text: str) -> str:
    text = normalize_text(value)
    if not text:
        return text
    context = (chunk_text or "").lower()
    sheep_context = "sheep" in context
    key = text.lower()
    if key == "sheep":
        return "sheep"
    if sheep_context and key in {"ewe", "ram", "lamb"}:
        return text
    if key in ENTITY_SINGULAR_MAP:
        return ENTITY_SINGULAR_MAP[key]
    return text


def _parse_json_array_response(raw: str) -> Tuple[List[Dict[str, Any]], bool, str]:
    cleaned = strip_think_tokens(raw or "").strip()
    if not cleaned:
        return [], False, "empty_response"

    payload = None
    last_error = ""
    try:
        payload = json.loads(cleaned)
    except Exception as exc:
        last_error = str(exc)
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception as inner_exc:
                last_error = str(inner_exc)
                payload = None

    if not isinstance(payload, list):
        return [], False, last_error or "response_is_not_json_array"

    triples: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            triples.append(
                {
                    "head": item.get("head", ""),
                    "relation": item.get("relation", ""),
                    "tail": item.get("tail", ""),
                }
            )
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            triples.append(
                {
                    "head": item[0],
                    "relation": item[1],
                    "tail": item[2],
                }
            )
    return triples, True, ""


def validate_triples(
    triples: Iterable[Dict[str, Any]],
    chunk_text: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    validated: List[Dict[str, str]] = []
    seen = set()
    relation_rewrite_count = 0
    direction_swap_count = 0
    filtered_supernode_triple_count = 0
    canonicalized_triples_count = 0
    relation_canonicalization_stats: Dict[str, int] = {}

    for triple in triples:
        head = _normalize_entity_text(triple.get("head", ""), chunk_text)
        relation = normalize_text(triple.get("relation", ""))
        tail = _normalize_entity_text(triple.get("tail", ""), chunk_text)

        if not head or not relation or not tail:
            continue

        relation_norm_raw = _normalize_graph_key(relation)
        relation_norm = RELATION_CANONICAL_MAP.get(relation_norm_raw, relation_norm_raw)
        if relation_norm != relation_norm_raw:
            relation_rewrite_count += 1
            canonicalized_triples_count += 1
            key = f"{relation_norm_raw}->{relation_norm}"
            relation_canonicalization_stats[key] = relation_canonicalization_stats.get(key, 0) + 1

        if relation_norm_raw == "caused_by":
            head, tail = tail, head
            direction_swap_count += 1
            canonicalized_triples_count += 1
            relation_norm = "causes"
            relation_canonicalization_stats["caused_by->causes_swap"] = (
                relation_canonicalization_stats.get("caused_by->causes_swap", 0) + 1
            )
        elif relation_norm_raw == "required_for":
            head, tail = tail, head
            direction_swap_count += 1
            canonicalized_triples_count += 1
            relation_norm = "requires"
            relation_canonicalization_stats["required_for->requires_swap"] = (
                relation_canonicalization_stats.get("required_for->requires_swap", 0) + 1
            )
        elif relation_norm_raw == "treated_by":
            relation_norm = "treated_with"

        head_norm = _normalize_graph_key(head)
        tail_norm = _normalize_graph_key(tail)

        if not head_norm or not relation_norm or not tail_norm:
            continue

        if (
            (head_norm in SUPER_NODE_NAME_NORMS or tail_norm in SUPER_NODE_NAME_NORMS)
            and relation_norm not in SUPER_NODE_ALLOW_RELATIONS
        ):
            filtered_supernode_triple_count += 1
            continue

        if head_norm == tail_norm:
            continue
        if len(relation_norm) < MIN_RELATION_LEN or len(relation_norm) > MAX_RELATION_LEN:
            continue
        if relation_norm in GENERIC_RELATIONS:
            continue

        key = (head_norm, relation_norm, tail_norm)
        if key in seen:
            continue
        seen.add(key)

        validated.append(
            {
                "head": head,
                "relation": relation_norm,
                "tail": tail,
                "head_norm": head_norm,
                "relation_norm": relation_norm,
                "tail_norm": tail_norm,
            }
        )

    stats = {
        "relation_rewrite_count": relation_rewrite_count,
        "direction_swap_count": direction_swap_count,
        "canonicalized_triples_count": canonicalized_triples_count,
        "filtered_supernode_triple_count": filtered_supernode_triple_count,
        "relation_canonicalization_stats": relation_canonicalization_stats,
    }
    return validated, stats


def extract_triples(
    client: Client,
    doc: Dict[str, str],
    model: str,
    language: str,
    retries: int = 2,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    del language

    prompt = TRIPLE_PROMPT_TEMPLATE.format(chunk=doc["text"])
    started = time.perf_counter()
    last_error = ""
    failed = False
    triples: List[Dict[str, str]] = []
    retry_count = 0
    raw_triples_count = 0
    validation_stats: Dict[str, Any] = {
        "relation_rewrite_count": 0,
        "direction_swap_count": 0,
        "canonicalized_triples_count": 0,
        "filtered_supernode_triple_count": 0,
        "relation_canonicalization_stats": {},
    }

    for attempt in range(retries + 1):
        try:
            response = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "top_p": 1.0},
            )
            content = response.get("message", {}).get("content", "")
            parsed, parse_ok, parse_error = _parse_json_array_response(content)
            if not parse_ok:
                last_error = f"json_parse_failed: {parse_error}"
                if attempt < retries:
                    retry_count += 1
                    continue
                failed = True
                break

            raw_triples_count = len(parsed)
            triples, validation_stats = validate_triples(parsed, doc.get("text", ""))
            last_error = ""
            failed = False
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt >= retries:
                failed = True
                break
            retry_count += 1

    latency_ms = int((time.perf_counter() - started) * 1000)
    observation = {
        "chunk_id": doc["id"],
        "hash": doc["hash"],
        "model": model,
        "raw_triples_count": raw_triples_count if not failed else 0,
        "valid_triples_count": len(triples),
        "canonicalized_triples_count": validation_stats.get("canonicalized_triples_count", 0) if not failed else 0,
        "filtered_triples_count": max(0, (raw_triples_count if not failed else 0) - len(triples)),
        "relation_rewrite_count": validation_stats.get("relation_rewrite_count", 0) if not failed else 0,
        "direction_swap_count": validation_stats.get("direction_swap_count", 0) if not failed else 0,
        "filtered_supernode_triple_count": validation_stats.get("filtered_supernode_triple_count", 0) if not failed else 0,
        "relation_canonicalization_stats": validation_stats.get("relation_canonicalization_stats", {}) if not failed else {},
        "num_triples": len(triples),
        "retry_count": retry_count,
        "latency_ms": latency_ms,
        "failed": failed,
        "error": last_error or None,
    }

    return triples, observation


def collect_triples_for_documents(
    client: Client,
    docs: List[Dict[str, str]],
    model: str,
    language: str,
) -> Tuple[Dict[str, List[Dict[str, str]]], List[str], List[str], List[Dict[str, Any]]]:
    triple_map: Dict[str, List[Dict[str, str]]] = {}
    empty_chunks: List[str] = []
    failed_chunks: List[str] = []
    observation_logs: List[Dict[str, Any]] = []

    if not docs:
        return triple_map, empty_chunks, failed_chunks, observation_logs

    max_workers = GENERATION_CONFIG.get("max_workers", 2)
    retries = GRAPH_INDEXING_CONFIG.get("triple_retries", GENERATION_CONFIG.get("max_retries", 2))
    print(f"  Starting deterministic triple extraction with {max_workers} workers")

    def process_doc(doc: Dict[str, str]) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        triples, observation = extract_triples(
            client=client,
            doc=doc,
            model=model,
            language=language,
            retries=retries,
        )
        return doc["id"], triples, observation

    total = len(docs)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {executor.submit(process_doc, doc): doc for doc in docs}

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                chunk_id, triples, observation = future.result(timeout=GENERATION_CONFIG.get("timeout", 150))
            except Exception as exc:
                chunk_id = doc["id"]
                triples = []
                observation = {
                    "chunk_id": chunk_id,
                    "hash": doc["hash"],
                    "model": model,
                    "num_triples": 0,
                    "retry_count": retries,
                    "latency_ms": 0,
                    "failed": True,
                    "error": str(exc),
                }

            observation_logs.append(observation)
            if observation["failed"]:
                failed_chunks.append(chunk_id)
            elif triples:
                triple_map[chunk_id] = triples
            else:
                empty_chunks.append(chunk_id)

            completed += 1
            progress = (completed / total) * 100
            print(
                f"\r  {completed}/{total} ({progress:.1f}%) - {chunk_id}: "
                f"{observation['num_triples']} triples, failed={observation['failed']}",
                end="",
                flush=True,
            )

    print()
    observation_logs.sort(key=lambda item: item["chunk_id"])
    return triple_map, empty_chunks, failed_chunks, observation_logs


def _cleanup_chunk_graph(session, chunk_id: str) -> None:
    session.run(
        """
        MATCH (c:Chunk {id:$cid})
        OPTIONAL MATCH (c)-[m:MENTIONS]->()
        DELETE m
        """,
        cid=chunk_id,
    )
    session.run(
        """
        MATCH ()-[r:RELATION]->()
        WHERE $cid IN coalesce(r.chunks, [])
        SET r.chunks = [chunk_id IN coalesce(r.chunks, []) WHERE chunk_id <> $cid]
        """,
        cid=chunk_id,
    )
    session.run(
        """
        MATCH ()-[r:RELATION]->()
        WHERE r.chunks IS NULL OR size(r.chunks) = 0
        DELETE r
        """
    )
    session.run(
        """
        MATCH (e:Entity)
        WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
          AND NOT (e)-[:RELATION]-()
        DELETE e
        """
    )


def _write_observation_jsonl(path: Path, run_metadata: Dict[str, Any], logs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for log in logs:
            record = dict(run_metadata)
            record.update(log)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def sample_empty_chunks(docs: List[Dict[str, str]], empty_chunks: List[str], n: int = 20) -> List[Dict[str, str]]:
    doc_map = {doc["id"]: doc for doc in docs}
    samples: List[Dict[str, str]] = []
    for chunk_id in empty_chunks[: max(0, n)]:
        doc = doc_map.get(chunk_id)
        if not doc:
            continue
        samples.append(
            {
                "id": doc["id"],
                "source": doc.get("source", ""),
                "preview": (doc.get("text", "") or "")[:500],
            }
        )
    return samples


def _compute_top_relation_types_from_logs(observation_logs: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    counter = Counter()
    for log in observation_logs:
        for relation in log.get("relation_types", []):
            counter[relation] += 1
    return [{"type_norm": rel, "count": count} for rel, count in counter.most_common(top_k)]


def audit_graph_quality(driver, limit: int = 30) -> Dict[str, Any]:
    with driver.session() as session:
        counts = session.run(
            """
            OPTIONAL MATCH (e:Entity)
            WITH count(e) AS entity_count
            OPTIONAL MATCH ()-[r:RELATION]->()
            WITH entity_count, count(r) AS relation_count
            OPTIONAL MATCH (c:Chunk)
            RETURN entity_count, relation_count, count(c) AS chunk_count
            """
        ).single()
        counts = counts or {"entity_count": 0, "relation_count": 0, "chunk_count": 0}
        entity_count = int(counts.get("entity_count") or 0)
        relation_count = int(counts.get("relation_count") or 0)
        chunk_count = int(counts.get("chunk_count") or 0)

        missing_entity_name_count = int(
            session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IS NULL OR trim(e.name) = ""
                RETURN count(e) AS cnt
                """
            ).single()["cnt"]
            or 0
        )
        missing_entity_name_norm_count = int(
            session.run(
                """
                MATCH (e:Entity)
                WHERE e.name_norm IS NULL OR trim(e.name_norm) = ""
                RETURN count(e) AS cnt
                """
            ).single()["cnt"]
            or 0
        )
        missing_relation_type_count = int(
            session.run(
                """
                MATCH ()-[r:RELATION]->()
                WHERE r.type IS NULL OR trim(r.type) = ""
                RETURN count(r) AS cnt
                """
            ).single()["cnt"]
            or 0
        )
        missing_relation_type_norm_count = int(
            session.run(
                """
                MATCH ()-[r:RELATION]->()
                WHERE r.type_norm IS NULL OR trim(r.type_norm) = ""
                RETURN count(r) AS cnt
                """
            ).single()["cnt"]
            or 0
        )
        missing_relation_chunks_count = int(
            session.run(
                """
                MATCH ()-[r:RELATION]->()
                WHERE r.chunks IS NULL OR size(r.chunks) = 0
                RETURN count(r) AS cnt
                """
            ).single()["cnt"]
            or 0
        )

        avg_degree = float(
            session.run(
                """
                MATCH (e:Entity)
                RETURN avg(COUNT { (e)-[:RELATION]-() }) AS avg_degree
                """
            ).single()["avg_degree"]
            or 0.0
        )
        degree_threshold = max(10.0, avg_degree * 3.0)
        degree_one_entity_count = int(
            session.run(
                """
                MATCH (e:Entity)
                WITH e, COUNT { (e)-[:RELATION]-() } AS degree
                WHERE degree = 1
                RETURN count(e) AS cnt
                """
            ).single()["cnt"]
            or 0
        )
        degree_one_entity_ratio = (float(degree_one_entity_count) / float(entity_count)) if entity_count > 0 else 0.0
        relation_chunk_stats = session.run(
            """
            MATCH ()-[r:RELATION]->()
            RETURN
                count(CASE WHEN size(coalesce(r.chunks, [])) > 1 THEN 1 END) AS multi_chunk_relation_count,
                avg(size(coalesce(r.chunks, []))) AS avg_chunks_per_relation,
                max(size(coalesce(r.chunks, []))) AS max_chunks_per_relation
            """
        ).single()
        multi_chunk_relation_count = int((relation_chunk_stats or {}).get("multi_chunk_relation_count") or 0)
        avg_chunks_per_relation = float((relation_chunk_stats or {}).get("avg_chunks_per_relation") or 0.0)
        max_chunks_per_relation = int((relation_chunk_stats or {}).get("max_chunks_per_relation") or 0)

        top_high_degree_entities = [
            {
                "name": row["name"],
                "name_norm": row["name_norm"],
                "degree": int(row["degree"] or 0),
            }
            for row in session.run(
                """
                MATCH (e:Entity)
                WITH e, COUNT { (e)-[:RELATION]-() } AS degree
                RETURN coalesce(e.name, "") AS name,
                       coalesce(e.name_norm, "") AS name_norm,
                       degree
                ORDER BY degree DESC, name_norm ASC
                LIMIT $limit
                """,
                limit=limit,
            )
        ]

        top_relation_types = [
            {"type": row["type"], "type_norm": row["type_norm"], "count": int(row["count"] or 0)}
            for row in session.run(
                """
                MATCH ()-[r:RELATION]->()
                RETURN coalesce(r.type, "") AS type,
                       coalesce(r.type_norm, "") AS type_norm,
                       count(r) AS count
                ORDER BY count DESC, type_norm ASC
                LIMIT $limit
                """,
                limit=limit,
            )
        ]

        possible_super_nodes: List[Dict[str, Any]] = []
        for item in top_high_degree_entities:
            is_generic_name = item["name_norm"] in SUPER_NODE_NAME_NORMS
            is_high_degree = float(item["degree"]) > degree_threshold
            if is_generic_name or is_high_degree:
                possible_super_nodes.append(
                    {
                        "name": item["name"],
                        "name_norm": item["name_norm"],
                        "degree": item["degree"],
                        "reasons": ",".join(
                            part
                            for part in (
                                "generic_name" if is_generic_name else "",
                                "high_degree" if is_high_degree else "",
                            )
                            if part
                        ),
                    }
                )

    return {
        "entity_count": entity_count,
        "relation_count": relation_count,
        "chunk_count": chunk_count,
        "degree_one_entity_count": degree_one_entity_count,
        "degree_one_entity_ratio": degree_one_entity_ratio,
        "multi_chunk_relation_count": multi_chunk_relation_count,
        "avg_chunks_per_relation": avg_chunks_per_relation,
        "max_chunks_per_relation": max_chunks_per_relation,
        "missing_entity_name_count": missing_entity_name_count,
        "missing_entity_name_norm_count": missing_entity_name_norm_count,
        "missing_relation_type_count": missing_relation_type_count,
        "missing_relation_type_norm_count": missing_relation_type_norm_count,
        "missing_relation_chunks_count": missing_relation_chunks_count,
        "top_high_degree_entities": top_high_degree_entities,
        "top_relation_types": top_relation_types,
        "possible_super_nodes": possible_super_nodes,
        "avg_degree": avg_degree,
        "high_degree_threshold": degree_threshold,
        "relation_canonicalization_stats": {},
        "filtered_supernode_triple_count": 0,
    }


def ingest_triples(
    driver,
    docs: List[Dict[str, str]],
    client: Client,
    model: str,
    language: str,
    run_metadata: Dict[str, Any],
) -> Tuple[int, int, List[str], List[str], List[Dict[str, Any]], int]:
    if not docs:
        return 0, 0, [], [], [], 0

    triple_map, empty_chunks, failed_chunks, observation_logs = collect_triples_for_documents(
        client=client,
        docs=docs,
        model=model,
        language=language,
    )

    updated = 0
    total_triples_input = 0
    run_id = run_metadata["run_id"]

    with driver.session() as session:
        for doc in docs:
            chunk_id = doc["id"]
            _cleanup_chunk_graph(session, chunk_id)
            triples = triple_map.get(chunk_id, [])
            if not triples:
                continue
            total_triples_input += len(triples)

            relation_types = sorted({triple["relation_norm"] for triple in triples})

            session.run(
                """
                UNWIND $triples AS triple
                MERGE (h:Entity {name_norm: triple.head_norm})
                ON CREATE SET
                    h.created_at = timestamp(),
                    h.name = triple.head
                SET
                    h.name = coalesce(h.name, triple.head),
                    h.display_name = coalesce(h.display_name, triple.head),
                    h.name_norm = triple.head_norm,
                    h.run_id = $run_id,
                    h.last_seen_at = timestamp()

                MERGE (t:Entity {name_norm: triple.tail_norm})
                ON CREATE SET
                    t.created_at = timestamp(),
                    t.name = triple.tail
                SET
                    t.name = coalesce(t.name, triple.tail),
                    t.display_name = coalesce(t.display_name, triple.tail),
                    t.name_norm = triple.tail_norm,
                    t.run_id = $run_id,
                    t.last_seen_at = timestamp()

                MERGE (h)-[r:RELATION {type_norm: triple.relation_norm}]->(t)
                ON CREATE SET
                    r.type = triple.relation,
                    r.type_norm = triple.relation_norm,
                    r.chunks = [$cid],
                    r.created_at = timestamp(),
                    r.run_id = $run_id,
                    r.graph_run_id = $run_id,
                    r.source_document_id = $source_document_id,
                    r.source_chunk_id = $cid,
                    r.source_type = $source_type,
                    r.source_question_id = $source_question_id,
                    r.category = $category,
                    r.supporting_quote = left($supporting_quote, 280),
                    r.extraction_model = $extraction_model,
                    r.confidence = $confidence
                ON MATCH SET
                    r.type = coalesce(r.type, triple.relation),
                    r.type_norm = triple.relation_norm,
                    r.chunks = CASE
                        WHEN $cid IN coalesce(r.chunks, []) THEN coalesce(r.chunks, [])
                        ELSE coalesce(r.chunks, []) + $cid
                    END,
                    r.last_updated = timestamp(),
                    r.run_id = $run_id,
                    r.graph_run_id = $run_id

                WITH h, t
                MATCH (c:Chunk {id:$cid})
                MERGE (c)-[mh:MENTIONS]->(h)
                ON CREATE SET mh.run_id = $run_id, mh.graph_run_id = $run_id, mh.source_chunk_id = $cid
                MERGE (c)-[mt:MENTIONS]->(t)
                ON CREATE SET mt.run_id = $run_id, mt.graph_run_id = $run_id, mt.source_chunk_id = $cid
                """,
                triples=triples,
                cid=chunk_id,
                run_id=run_id,
                source_document_id=doc.get("document_id") or chunk_id,
                source_type=doc.get("source_type", "unknown"),
                source_question_id=doc.get("source_question_id"),
                category=doc.get("category"),
                supporting_quote=(doc.get("text", "") or "").replace("\n", " "),
                extraction_model=model,
                confidence=1.0,
            )
            updated += 1
            for item in observation_logs:
                if item["chunk_id"] == chunk_id:
                    item["relation_types"] = relation_types
                    break

        session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
              AND NOT (e)-[:RELATION]-()
            DELETE e
            """
        )

    skipped = len(docs) - updated
    return updated, skipped, empty_chunks, failed_chunks, observation_logs, total_triples_input


def _resolve_graph_create_model(requested_model: str) -> str:
    model_aliases = {
        "deepseekr1-14b-fp16": ["deepseek-r1:14b", "deepseek-r1:14b-qwen-distill-q4_K_M"],
    }
    preferred = [requested_model] + model_aliases.get(requested_model, [])
    for name in preferred:
        try:
            rc = os.system(f'ollama show "{name}" >nul 2>nul')
            if rc == 0:
                return name
        except Exception:
            continue
    return requested_model


class GraphBuilder:
    def __init__(self, driver, ollama_client: Client):
        self.driver = driver
        self.client = ollama_client
        self.embedder = OllamaVectorEmbedder(self.client, CONFIG["models"]["embed_model"])

    def build_graph(
        self,
        text_path: Path,
        chunk_size: int = None,
        overlap: int = None,
        setting: str = "kg_corpus_only",
        qaset_path: Path | None = None,
        directqa_ids: List[int] | None = None,
        indirectqa_ids_excluded_from_index: List[int] | None = None,
    ):
        effective_chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        effective_overlap = overlap if overlap is not None else DEFAULT_CHUNK_OVERLAP
        requested_graph_model = CONFIG["models"]["graph_create_model"]
        resolved_graph_model = _resolve_graph_create_model(requested_graph_model)
        if resolved_graph_model != requested_graph_model:
            print(f"Graph create model fallback: {requested_graph_model} -> {resolved_graph_model}")

        run_metadata = {
            "run_id": str(uuid.uuid4()),
            "dataset_id": DATASET_ID,
            "chunk_size": effective_chunk_size,
            "overlap": effective_overlap,
            "embed_model": CONFIG["models"]["embed_model"],
            "graph_create_model": requested_graph_model,
            "graph_create_model_resolved": resolved_graph_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        snapshot = GraphSnapshotManager(neo4j_database=CONFIG["infrastructure"].get("neo4j_database", "neo4j"))
        run_ctx = snapshot.start_run(setting=setting)
        # force run_id to snapshot id for provenance consistency
        run_metadata["run_id"] = run_ctx.graph_run_id

        print("Loading and chunking...")
        chunks = load_chunks(text_path, effective_chunk_size, effective_overlap)
        print(f"  Loaded {len(chunks)} chunks")

        print("Ensuring indexes...")
        ensure_entity_index(self.driver)
        ensure_graph_run_index(self.driver)
        ensure_vector_index(
            self.driver,
            CONFIG["infrastructure"]["vector_index_name"],
            "Chunk",
            "embedding",
            self.embedder.dimension,
        )
        ensure_fulltext_index(
            self.driver,
            CONFIG["infrastructure"]["fulltext_index_name"],
            "Chunk",
            "text",
        )

        print("Upserting chunks...")
        changed_docs, skipped_docs = upsert_chunks(
            self.driver,
            self.embedder,
            chunks,
            run_id=run_metadata["run_id"],
        )
        print(f"  Changed {len(changed_docs)} chunks, skipped {len(skipped_docs)} unchanged chunks")

        print("Extracting and ingesting triples for changed chunks only...")
        updated, skipped_triples, empty_chunks, failed_chunks, observation_logs, total_triples_input = ingest_triples(
            self.driver,
            changed_docs,
            self.client,
            resolved_graph_model,
            language=CONFIG["models"]["answer_language"],
            run_metadata=run_metadata,
        )
        print(
            f"  Triple-updated {updated} chunks, skipped {skipped_triples}, "
            f"empty {len(empty_chunks)}, failed {len(failed_chunks)}"
        )

        quality_audit = audit_graph_quality(self.driver)
        relation_canonicalization_stats: Dict[str, int] = {}
        filtered_supernode_triple_count = 0
        for log in observation_logs:
            filtered_supernode_triple_count += int(log.get("filtered_supernode_triple_count", 0) or 0)
            for key, value in (log.get("relation_canonicalization_stats", {}) or {}).items():
                relation_canonicalization_stats[key] = relation_canonicalization_stats.get(key, 0) + int(value)
        quality_audit["relation_canonicalization_stats"] = relation_canonicalization_stats
        quality_audit["filtered_supernode_triple_count"] = filtered_supernode_triple_count

        avg_extraction_latency_ms = (
            sum(int(item.get("latency_ms", 0)) for item in observation_logs) / len(observation_logs)
            if observation_logs
            else 0.0
        )
        avg_triples_per_updated_chunk = (float(total_triples_input) / float(updated)) if updated > 0 else 0.0

        summary = {
            "total_triples": quality_audit["relation_count"],
            "unique_entities": quality_audit["entity_count"],
            "unique_relations": quality_audit["relation_count"],
            "avg_triples_per_updated_chunk": avg_triples_per_updated_chunk,
            "avg_extraction_latency_ms": avg_extraction_latency_ms,
            "top_relation_types": quality_audit["top_relation_types"],
            "top_high_degree_entities": quality_audit["top_high_degree_entities"],
        }

        if GRAPH_INDEXING_CONFIG.get("write_observation_jsonl"):
            out_dir = Path(GRAPH_INDEXING_CONFIG["observation_log_dir"])
            run_id = run_metadata["run_id"]
            jsonl_path = out_dir / f"{run_id}.jsonl"
            summary_path = out_dir / f"{run_id}.summary.json"
            _write_observation_jsonl(jsonl_path, run_metadata, observation_logs)
            _write_json(
                summary_path,
                {
                    "run_metadata": run_metadata,
                    "stats": {
                        "total_chunks": len(chunks),
                        "changed_chunks": len(changed_docs),
                        "unchanged_chunks": len(skipped_docs),
                        "updated_chunks": updated,
                        "skipped_triple_ingestion_chunks": skipped_triples,
                        "empty_chunk_count": len(empty_chunks),
                        "failed_chunk_count": len(failed_chunks),
                    },
                    "summary": summary,
                    "quality_audit": quality_audit,
                    "empty_chunk_samples": sample_empty_chunks(changed_docs, empty_chunks, n=20),
                },
            )
            print(f"  Observation logs written to {jsonl_path}")
            print(f"  Run summary written to {summary_path}")

        # Snapshot export
        build_log_lines = [
            f"# Graph Build Log",
            f"- graph_run_id: {run_ctx.graph_run_id}",
            f"- setting: {setting}",
            f"- created_at: {run_ctx.created_at}",
            f"- dataset_id: {DATASET_ID}",
            f"- chunk_size: {effective_chunk_size}",
            f"- overlap: {effective_overlap}",
            f"- changed_chunks: {len(changed_docs)}",
            f"- updated_chunks: {updated}",
            f"- entity_count: {quality_audit['entity_count']}",
            f"- relation_count: {quality_audit['relation_count']}",
        ]
        snapshot.write_build_log(run_ctx, build_log_lines)
        artifact_status = snapshot.export_subgraph_artifacts(self.driver, run_ctx)
        graphml_status = snapshot.build_graphml(run_ctx)
        dump_status = snapshot.try_neo4j_dump(run_ctx)
        if dump_status.get("status") != "ok":
            artifact_status["neo4j.dump"] = {"status": "failed", "reason": dump_status}
        else:
            artifact_status["neo4j.dump"] = {"status": "ok", "path": dump_status.get("path")}

        corpus_sha = sha256_file(text_path) if text_path.exists() else None
        qaset_sha = sha256_file(qaset_path) if qaset_path and qaset_path.exists() else None
        manifest = {
            "graph_run_id": run_ctx.graph_run_id,
            "setting": setting,
            "created_at": run_ctx.created_at,
            "artifact_dir": str(run_ctx.artifact_dir),
            "neo4j_database": run_ctx.neo4j_database,
            "neo4j_dump_status": dump_status.get("status"),
            "neo4j_dump_path": dump_status.get("path"),
            "corpus_sha256": corpus_sha,
            "qaset_sha256": qaset_sha,
            "directqa_ids": directqa_ids or [],
            "indirectqa_ids_excluded_from_index": indirectqa_ids_excluded_from_index or [],
            "extraction_model": resolved_graph_model,
            "embedding_model": CONFIG["models"]["embed_model"],
            "chunk_size": effective_chunk_size,
            "chunk_overlap": effective_overlap,
            "entity_count": quality_audit["entity_count"],
            "relation_count": quality_audit["relation_count"],
            "claim_count": artifact_status.get("claims.jsonl", {}).get("count", 0),
            "community_count": artifact_status.get("communities.jsonl", {}).get("count", 0),
            "status": "completed",
            "artifact_status": {
                **artifact_status,
                "graph.graphml": graphml_status,
                "vector_index": {"status": "unavailable"},
                "bm25_index": {"status": "unavailable"},
            },
        }
        snapshot.write_manifest(run_ctx, manifest)
        snapshot.append_registry(
            {
                "graph_run_id": run_ctx.graph_run_id,
                "setting": setting,
                "created_at": run_ctx.created_at,
                "artifact_dir": str(run_ctx.artifact_dir),
                "neo4j_dump_path": dump_status.get("path"),
                "status": manifest["status"],
                "counts": {
                    "chunks": len(chunks),
                    "entities": quality_audit["entity_count"],
                    "relations": quality_audit["relation_count"],
                    "claims": manifest["claim_count"],
                    "communities": manifest["community_count"],
                },
                "source_hashes": {"corpus_sha256": corpus_sha, "qaset_sha256": qaset_sha},
                "used_for_eval_runs": [],
            }
        )
        snapshot.create_graphrun_node(
            self.driver,
            {
                "graph_run_id": run_ctx.graph_run_id,
                "setting": setting,
                "created_at": run_ctx.created_at,
                "artifact_dir": str(run_ctx.artifact_dir),
                "neo4j_dump_path": dump_status.get("path"),
                "corpus_sha256": corpus_sha,
                "qaset_sha256": qaset_sha,
                "directqa_ids": directqa_ids or [],
                "indirectqa_ids_excluded_from_index": indirectqa_ids_excluded_from_index or [],
                "extraction_model": resolved_graph_model,
                "embedding_model": CONFIG["models"]["embed_model"],
                "chunk_size": effective_chunk_size,
                "chunk_overlap": effective_overlap,
                "entity_count": quality_audit["entity_count"],
                "relation_count": quality_audit["relation_count"],
                "claim_count": manifest["claim_count"],
                "community_count": manifest["community_count"],
                "status": "completed",
            },
            neo4j_database=run_ctx.neo4j_database,
        )

        result = {
            "run_metadata": run_metadata,
            "graph_run_id": run_ctx.graph_run_id,
            "artifact_dir": str(run_ctx.artifact_dir),
            "neo4j_database": run_ctx.neo4j_database,
            "changed_docs": changed_docs,
            "skipped_docs": skipped_docs,
            "empty_chunks": empty_chunks,
            "failed_chunks": failed_chunks,
            "observation_logs": observation_logs,
            "summary": summary,
            "quality_audit": quality_audit,
            "empty_chunk_samples": sample_empty_chunks(changed_docs, empty_chunks, n=20),
            "stats": {
                "total_chunks": len(chunks),
                "changed_chunks": len(changed_docs),
                "unchanged_chunks": len(skipped_docs),
                "updated_chunks": updated,
                "skipped_triple_ingestion_chunks": skipped_triples,
                "empty_chunk_count": len(empty_chunks),
                "failed_chunk_count": len(failed_chunks),
            },
        }

        print("Graph build completed")
        return result

    def build_graph_from_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = None,
        overlap: int = None,
        setting: str = "kg_custom_docs",
        qaset_path: Path | None = None,
        directqa_ids: List[int] | None = None,
        indirectqa_ids_excluded_from_index: List[int] | None = None,
    ):
        # Reuse the same deterministic pipeline while preserving per-document provenance.
        tmp_path = Path("data/tmp_graph_build_documents_snapshot.txt")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            "\n\n".join(str(d.get("text", "")).strip() for d in documents if str(d.get("text", "")).strip()),
            encoding="utf-8",
        )

        effective_chunk_size = chunk_size if chunk_size is not None else DEFAULT_CHUNK_SIZE
        effective_overlap = overlap if overlap is not None else DEFAULT_CHUNK_OVERLAP
        requested_graph_model = CONFIG["models"]["graph_create_model"]
        resolved_graph_model = _resolve_graph_create_model(requested_graph_model)
        if resolved_graph_model != requested_graph_model:
            print(f"Graph create model fallback: {requested_graph_model} -> {resolved_graph_model}")

        run_metadata = {
            "run_id": str(uuid.uuid4()),
            "dataset_id": DATASET_ID,
            "chunk_size": effective_chunk_size,
            "overlap": effective_overlap,
            "embed_model": CONFIG["models"]["embed_model"],
            "graph_create_model": requested_graph_model,
            "graph_create_model_resolved": resolved_graph_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        snapshot = GraphSnapshotManager(neo4j_database=CONFIG["infrastructure"].get("neo4j_database", "neo4j"))
        run_ctx = snapshot.start_run(setting=setting)
        run_metadata["run_id"] = run_ctx.graph_run_id

        print("Loading and chunking from provided documents...")
        chunks = load_chunks_from_documents(documents, effective_chunk_size, effective_overlap)
        print(f"  Loaded {len(chunks)} chunks from {len(documents)} documents")

        print("Ensuring indexes...")
        ensure_entity_index(self.driver)
        ensure_graph_run_index(self.driver)
        ensure_vector_index(
            self.driver,
            CONFIG["infrastructure"]["vector_index_name"],
            "Chunk",
            "embedding",
            self.embedder.dimension,
        )
        ensure_fulltext_index(
            self.driver,
            CONFIG["infrastructure"]["fulltext_index_name"],
            "Chunk",
            "text",
        )

        print("Upserting chunks...")
        changed_docs, skipped_docs = upsert_chunks(
            self.driver,
            self.embedder,
            chunks,
            run_id=run_metadata["run_id"],
        )
        print(f"  Changed {len(changed_docs)} chunks, skipped {len(skipped_docs)} unchanged chunks")

        print("Extracting and ingesting triples for changed chunks only...")
        updated, skipped_triples, empty_chunks, failed_chunks, observation_logs, total_triples_input = ingest_triples(
            self.driver,
            changed_docs,
            self.client,
            resolved_graph_model,
            language=CONFIG["models"]["answer_language"],
            run_metadata=run_metadata,
        )
        print(
            f"  Triple-updated {updated} chunks, skipped {skipped_triples}, "
            f"empty {len(empty_chunks)}, failed {len(failed_chunks)}"
        )

        quality_audit = audit_graph_quality(self.driver)
        relation_canonicalization_stats: Dict[str, int] = {}
        filtered_supernode_triple_count = 0
        for log in observation_logs:
            filtered_supernode_triple_count += int(log.get("filtered_supernode_triple_count", 0) or 0)
            for key, value in (log.get("relation_canonicalization_stats", {}) or {}).items():
                relation_canonicalization_stats[key] = relation_canonicalization_stats.get(key, 0) + int(value)
        quality_audit["relation_canonicalization_stats"] = relation_canonicalization_stats
        quality_audit["filtered_supernode_triple_count"] = filtered_supernode_triple_count

        avg_extraction_latency_ms = (
            sum(int(item.get("latency_ms", 0)) for item in observation_logs) / len(observation_logs)
            if observation_logs
            else 0.0
        )
        avg_triples_per_updated_chunk = (float(total_triples_input) / float(updated)) if updated > 0 else 0.0

        summary = {
            "total_triples": quality_audit["relation_count"],
            "unique_entities": quality_audit["entity_count"],
            "unique_relations": quality_audit["relation_count"],
            "avg_triples_per_updated_chunk": avg_triples_per_updated_chunk,
            "avg_extraction_latency_ms": avg_extraction_latency_ms,
            "top_relation_types": quality_audit["top_relation_types"],
            "top_high_degree_entities": quality_audit["top_high_degree_entities"],
        }

        if GRAPH_INDEXING_CONFIG.get("write_observation_jsonl"):
            out_dir = Path(GRAPH_INDEXING_CONFIG["observation_log_dir"])
            run_id = run_metadata["run_id"]
            jsonl_path = out_dir / f"{run_id}.jsonl"
            summary_path = out_dir / f"{run_id}.summary.json"
            _write_observation_jsonl(jsonl_path, run_metadata, observation_logs)
            _write_json(
                summary_path,
                {
                    "run_metadata": run_metadata,
                    "stats": {
                        "total_chunks": len(chunks),
                        "changed_chunks": len(changed_docs),
                        "unchanged_chunks": len(skipped_docs),
                        "updated_chunks": updated,
                        "skipped_triple_ingestion_chunks": skipped_triples,
                        "empty_chunk_count": len(empty_chunks),
                        "failed_chunk_count": len(failed_chunks),
                    },
                    "summary": summary,
                    "quality_audit": quality_audit,
                    "empty_chunk_samples": sample_empty_chunks(changed_docs, empty_chunks, n=20),
                },
            )

        build_log_lines = [
            "# Graph Build Log",
            f"- graph_run_id: {run_ctx.graph_run_id}",
            f"- setting: {setting}",
            f"- created_at: {run_ctx.created_at}",
            f"- dataset_id: {DATASET_ID}",
            f"- chunk_size: {effective_chunk_size}",
            f"- overlap: {effective_overlap}",
            f"- changed_chunks: {len(changed_docs)}",
            f"- updated_chunks: {updated}",
            f"- entity_count: {quality_audit['entity_count']}",
            f"- relation_count: {quality_audit['relation_count']}",
        ]
        snapshot.write_build_log(run_ctx, build_log_lines)
        artifact_status = snapshot.export_subgraph_artifacts(self.driver, run_ctx)
        graphml_status = snapshot.build_graphml(run_ctx)
        dump_status = snapshot.try_neo4j_dump(run_ctx)
        if dump_status.get("status") != "ok":
            artifact_status["neo4j.dump"] = {"status": "failed", "reason": dump_status}
        else:
            artifact_status["neo4j.dump"] = {"status": "ok", "path": dump_status.get("path")}

        corpus_sha = sha256_file(tmp_path) if tmp_path.exists() else None
        qaset_sha = sha256_file(qaset_path) if qaset_path and qaset_path.exists() else None
        manifest = {
            "graph_run_id": run_ctx.graph_run_id,
            "setting": setting,
            "created_at": run_ctx.created_at,
            "artifact_dir": str(run_ctx.artifact_dir),
            "neo4j_database": run_ctx.neo4j_database,
            "neo4j_dump_status": dump_status.get("status"),
            "neo4j_dump_path": dump_status.get("path"),
            "corpus_sha256": corpus_sha,
            "qaset_sha256": qaset_sha,
            "directqa_ids": directqa_ids or [],
            "indirectqa_ids_excluded_from_index": indirectqa_ids_excluded_from_index or [],
            "extraction_model": resolved_graph_model,
            "embedding_model": CONFIG["models"]["embed_model"],
            "chunk_size": effective_chunk_size,
            "chunk_overlap": effective_overlap,
            "entity_count": quality_audit["entity_count"],
            "relation_count": quality_audit["relation_count"],
            "claim_count": artifact_status.get("claims.jsonl", {}).get("count", 0),
            "community_count": artifact_status.get("communities.jsonl", {}).get("count", 0),
            "status": "completed",
            "artifact_status": {
                **artifact_status,
                "graph.graphml": graphml_status,
                "vector_index": {"status": "unavailable"},
                "bm25_index": {"status": "unavailable"},
            },
        }
        snapshot.write_manifest(run_ctx, manifest)
        snapshot.append_registry(
            {
                "graph_run_id": run_ctx.graph_run_id,
                "setting": setting,
                "created_at": run_ctx.created_at,
                "artifact_dir": str(run_ctx.artifact_dir),
                "neo4j_dump_path": dump_status.get("path"),
                "status": manifest["status"],
                "counts": {
                    "chunks": len(chunks),
                    "entities": quality_audit["entity_count"],
                    "relations": quality_audit["relation_count"],
                    "claims": manifest["claim_count"],
                    "communities": manifest["community_count"],
                },
                "source_hashes": {"corpus_sha256": corpus_sha, "qaset_sha256": qaset_sha},
                "used_for_eval_runs": [],
            }
        )
        snapshot.create_graphrun_node(
            self.driver,
            {
                "graph_run_id": run_ctx.graph_run_id,
                "setting": setting,
                "created_at": run_ctx.created_at,
                "artifact_dir": str(run_ctx.artifact_dir),
                "neo4j_dump_path": dump_status.get("path"),
                "corpus_sha256": corpus_sha,
                "qaset_sha256": qaset_sha,
                "directqa_ids": directqa_ids or [],
                "indirectqa_ids_excluded_from_index": indirectqa_ids_excluded_from_index or [],
                "extraction_model": resolved_graph_model,
                "embedding_model": CONFIG["models"]["embed_model"],
                "chunk_size": effective_chunk_size,
                "chunk_overlap": effective_overlap,
                "entity_count": quality_audit["entity_count"],
                "relation_count": quality_audit["relation_count"],
                "claim_count": manifest["claim_count"],
                "community_count": manifest["community_count"],
                "status": "completed",
            },
            neo4j_database=run_ctx.neo4j_database,
        )

        result = {
            "run_metadata": run_metadata,
            "graph_run_id": run_ctx.graph_run_id,
            "artifact_dir": str(run_ctx.artifact_dir),
            "neo4j_database": run_ctx.neo4j_database,
            "changed_docs": changed_docs,
            "skipped_docs": skipped_docs,
            "empty_chunks": empty_chunks,
            "failed_chunks": failed_chunks,
            "observation_logs": observation_logs,
            "summary": summary,
            "quality_audit": quality_audit,
            "empty_chunk_samples": sample_empty_chunks(changed_docs, empty_chunks, n=20),
            "stats": {
                "total_chunks": len(chunks),
                "changed_chunks": len(changed_docs),
                "unchanged_chunks": len(skipped_docs),
                "updated_chunks": updated,
                "skipped_triple_ingestion_chunks": skipped_triples,
                "empty_chunk_count": len(empty_chunks),
                "failed_chunk_count": len(failed_chunks),
            },
        }
        print("Graph build completed (documents mode)")
        return result
