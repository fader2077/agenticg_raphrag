"""Neo4j-backed HotpotQA train graph builder using neo4j-graphrag indexes."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import DownloadConfig, load_dataset
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_fulltext_index

from src.data.schema import ChunkSpec, doc_id_for_title, make_chunks, normalize_title
from src.graph.canonicalization import entity_key
from src.graph.relation_extraction import extract_relations_from_chunk, extract_relations_from_chunk_ollama, merge_relation_edges
from src.io_utils import append_jsonl, read_jsonl, write_json, write_jsonl


DEFAULT_GRAPH_RUN_ID = "hotpotqa_train_full_neo4j_v1"
DEFAULT_FULLTEXT_INDEX_NAME = "hotpotqa_train_hotpotchunk_text_fts_v1"
DEFAULT_ENTITY_FULLTEXT_INDEX_NAME = "hotpotqa_train_hotpotentity_text_fts_v1"


def neo4j_auth_from_env() -> tuple[str, tuple[str, str]]:
    """Return Neo4j URI and auth tuple from environment/config defaults."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "neo4jgoat"))
    return uri, auth


def graph_exists(driver: Any, graph_run_id: str) -> dict[str, Any]:
    """Return graph state counts for a graph_run_id."""
    with driver.session() as session:
        rec = session.run(
            """
            MATCH (g:GraphRun {graph_run_id:$gid})
            OPTIONAL MATCH (c:HotpotChunk {graph_run_id:$gid})
            WITH g, count(c) AS chunks
            OPTIONAL MATCH (e:HotpotEntity {graph_run_id:$gid})
            WITH g, chunks, count(e) AS entities
            OPTIONAL MATCH (:HotpotEntity)-[r:HOTPOT_RELATION {graph_run_id:$gid}]->(:HotpotEntity)
            RETURN g.status AS status, chunks, entities, count(r) AS relations
            """,
            gid=graph_run_id,
        ).single()
    if not rec:
        return {"exists": False, "status": None, "chunks": 0, "entities": 0, "relations": 0}
    return {
        "exists": bool(rec.get("status")),
        "status": rec.get("status"),
        "chunks": int(rec.get("chunks") or 0),
        "entities": int(rec.get("entities") or 0),
        "relations": int(rec.get("relations") or 0),
    }


def _quote_index_name(name: str) -> str:
    """Return a safely quoted Neo4j index name."""
    return "`" + str(name).replace("`", "``") + "`"


def create_hotpot_fulltext_index(driver: Any, fulltext_index_name: str) -> dict[str, Any]:
    """Create the HotpotQA fulltext index through neo4j-graphrag."""
    status: dict[str, Any] = {"neo4j_graphrag_fulltext": "not_attempted"}
    try:
        create_fulltext_index(driver, fulltext_index_name, "HotpotChunk", ["text", "title"], fail_if_exists=False)
        status["neo4j_graphrag_fulltext"] = "ok"
    except Exception as exc:
        status["neo4j_graphrag_fulltext"] = f"failed:{type(exc).__name__}:{exc}"
    return status


def create_hotpot_entity_fulltext_index(driver: Any, entity_index_name: str = DEFAULT_ENTITY_FULLTEXT_INDEX_NAME) -> dict[str, Any]:
    """Create the HotpotQA entity fulltext index for fast entity linking."""
    status: dict[str, Any] = {"neo4j_graphrag_entity_fulltext": "not_attempted"}
    try:
        create_fulltext_index(driver, entity_index_name, "HotpotEntity", ["name", "display_name", "name_norm"], fail_if_exists=False)
        status["neo4j_graphrag_entity_fulltext"] = "ok"
    except Exception as exc:
        status["neo4j_graphrag_entity_fulltext"] = f"failed:{type(exc).__name__}:{exc}"
    return status


def drop_hotpot_fulltext_index(driver: Any, fulltext_index_name: str) -> dict[str, Any]:
    """Drop the HotpotQA fulltext index before bulk writes to avoid per-row index churn."""
    status: dict[str, Any] = {"dropped_fulltext_before_bulk_load": False}
    try:
        with driver.session() as session:
            session.run(f"DROP INDEX {_quote_index_name(fulltext_index_name)} IF EXISTS")
            session.run(f"DROP INDEX {_quote_index_name(DEFAULT_ENTITY_FULLTEXT_INDEX_NAME)} IF EXISTS")
        status["dropped_fulltext_before_bulk_load"] = True
    except Exception as exc:
        status["drop_fulltext_error"] = f"{type(exc).__name__}:{exc}"
    return status


def ensure_schema(driver: Any) -> dict[str, Any]:
    """Ensure HotpotQA graph constraints and lookup indexes exist."""
    status: dict[str, Any] = {"schema": "not_attempted"}
    with driver.session() as session:
        session.run("CREATE CONSTRAINT hotpotqa_graph_run IF NOT EXISTS FOR (g:GraphRun) REQUIRE g.graph_run_id IS UNIQUE")
        session.run("CREATE CONSTRAINT hotpotqa_hotpotchunk_key IF NOT EXISTS FOR (c:HotpotChunk) REQUIRE c.chunk_key IS UNIQUE")
        session.run("CREATE INDEX hotpotqa_hotpotchunk_id IF NOT EXISTS FOR (c:HotpotChunk) ON (c.id)")
        session.run("CREATE CONSTRAINT hotpotqa_hotpotentity_key IF NOT EXISTS FOR (e:HotpotEntity) REQUIRE e.entity_key IS UNIQUE")
        session.run("CREATE CONSTRAINT hotpotqa_hotpot_relation_id IF NOT EXISTS FOR ()-[r:HOTPOT_RELATION]-() REQUIRE r.relation_id IS UNIQUE")
        session.run("CREATE INDEX hotpotqa_hotpotentity_name_norm IF NOT EXISTS FOR (e:HotpotEntity) ON (e.name_norm)")
        session.run("CREATE INDEX hotpotqa_hotpot_relation_run IF NOT EXISTS FOR ()-[r:HOTPOT_RELATION]-() ON (r.graph_run_id)")
        session.run("CREATE INDEX hotpotqa_hotpotchunk_run IF NOT EXISTS FOR (c:HotpotChunk) ON (c.graph_run_id)")
        session.run("CREATE INDEX hotpotqa_hotpotentity_run IF NOT EXISTS FOR (e:HotpotEntity) ON (e.graph_run_id)")
    status["schema"] = "ok"
    return status


def delete_graph_run(driver: Any, graph_run_id: str) -> None:
    """Delete an existing graph run from Neo4j."""
    with driver.session() as session:
        session.run("MATCH (:HotpotEntity)-[r:HOTPOT_RELATION {graph_run_id:$gid}]->(:HotpotEntity) DELETE r", gid=graph_run_id)
        session.run("MATCH (:HotpotChunk {graph_run_id:$gid})-[r:HOTPOT_MENTIONS]->(:HotpotEntity) DELETE r", gid=graph_run_id)
        session.run("MATCH (c:HotpotChunk {graph_run_id:$gid}) DELETE c", gid=graph_run_id)
        session.run("MATCH (e:HotpotEntity {graph_run_id:$gid}) DELETE e", gid=graph_run_id)
        session.run("MATCH (:Entity)-[r:RELATION {graph_run_id:$gid}]->(:Entity) DELETE r", gid=graph_run_id)
        session.run("MATCH (:Chunk {graph_run_id:$gid})-[r:MENTIONS]->(:Entity) DELETE r", gid=graph_run_id)
        session.run("MATCH (c:Chunk {graph_run_id:$gid}) DELETE c", gid=graph_run_id)
        session.run("MATCH (g:GraphRun {graph_run_id:$gid}) DELETE g", gid=graph_run_id)


def iter_unique_train_chunks(
    split: str = "train",
    chunk_spec: ChunkSpec | None = None,
    max_docs: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield sentence-aware chunks for every unique HotpotQA document in a split."""
    spec = chunk_spec or ChunkSpec(max_sentences_per_chunk=3, overlap_sentences=1)
    try:
        ds = load_dataset("hotpot_qa", "distractor", split=split, download_config=DownloadConfig(local_files_only=True))
    except Exception:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
    seen_doc_ids: set[str] = set()
    yielded_docs = 0
    for row in ds:
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentence_groups = context.get("sentences") or []
        for title, sentences in zip(titles, sentence_groups):
            clean_title = normalize_title(title)
            doc_id = doc_id_for_title(clean_title)
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            document = {
                "title": clean_title,
                "sentences": [str(s).strip() for s in sentences if str(s).strip()],
                "doc_id": doc_id,
            }
            for chunk in make_chunks(document, spec):
                yield chunk
            yielded_docs += 1
            if max_docs is not None and yielded_docs >= max_docs:
                return


def _chunk_row(chunk: dict[str, Any], graph_run_id: str) -> dict[str, Any]:
    return {
        "chunk_key": f"{graph_run_id}::{chunk['chunk_id']}",
        "id": chunk["chunk_id"],
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "text": chunk["text"],
        "sentence_ids": list(chunk.get("sentence_ids", [])),
        "sentences": list(chunk.get("sentences", [])),
        "source": "hotpotqa_train",
        "dataset": "hotpotqa",
        "split": "train",
        "graph_run_id": graph_run_id,
    }


def _edge_rows(chunk: dict[str, Any], graph_run_id: str, max_edges_per_chunk: int) -> list[dict[str, Any]]:
    edges = extract_relations_from_chunk(chunk, max_entities=3, max_edges=max_edges_per_chunk)
    rows = []
    for idx, edge in enumerate(edges):
        head_key = entity_key(edge["head"])
        tail_key = entity_key(edge["tail"])
        if not head_key or not tail_key or head_key == tail_key:
            continue
        relation = str(edge.get("relation") or "co_occurs_with")
        relation_id = f"{graph_run_id}::{head_key}::{relation}::{tail_key}::{edge['source_chunk_id']}::{idx}"
        rows.append(
            {
                "relation_id": relation_id,
                "head": edge["head"],
                "head_norm": head_key,
                "head_key": f"{graph_run_id}::{head_key}",
                "tail": edge["tail"],
                "tail_norm": tail_key,
                "tail_key": f"{graph_run_id}::{tail_key}",
                "type": relation,
                "type_norm": relation,
                "source_doc_id": edge.get("source_doc_id"),
                "source_document_id": edge.get("source_doc_id"),
                "source_title": edge.get("source_title"),
                "source_chunk_id": edge.get("source_chunk_id"),
                "source_sentence_ids": list(edge.get("source_sentence_ids", [])),
                "supporting_quote": edge.get("supporting_quote", ""),
                "extractor": edge.get("extractor", "rule_based_titlecase_comention_v1"),
                "confidence": float(edge.get("confidence", 0.6) or 0.6),
                "graph_run_id": graph_run_id,
                "run_id": graph_run_id,
            }
        )
    return rows


def _slices(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield bounded list slices for Neo4j transactions."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _write_batch(driver: Any, chunks: list[dict[str, Any]], edges: list[dict[str, Any]], graph_run_id: str) -> None:
    """Write one chunk/edge batch to Neo4j."""
    with driver.session() as session:
        if chunks:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (c:HotpotChunk {chunk_key: row.chunk_key})
                SET c.id = row.id,
                    c.chunk_id = row.chunk_id,
                    c.chunk_key = row.chunk_key,
                    c.doc_id = row.doc_id,
                    c.source_document_id = row.doc_id,
                    c.title = row.title,
                    c.text = row.text,
                    c.sentence_ids = row.sentence_ids,
                    c.sentences = row.sentences,
                    c.source = row.source,
                    c.dataset = row.dataset,
                    c.split = row.split,
                    c.graph_run_id = row.graph_run_id,
                    c.run_id = row.graph_run_id,
                    c.updated_at = timestamp()
                """,
                rows=chunks,
            )
        if edges:
            entity_rows: dict[str, dict[str, Any]] = {}
            for edge in edges:
                entity_rows[edge["head_key"]] = {
                    "entity_key": edge["head_key"],
                    "name": edge["head"],
                    "name_norm": edge["head_norm"],
                    "graph_run_id": edge["graph_run_id"],
                }
                entity_rows[edge["tail_key"]] = {
                    "entity_key": edge["tail_key"],
                    "name": edge["tail"],
                    "name_norm": edge["tail_norm"],
                    "graph_run_id": edge["graph_run_id"],
                }
            session.run(
                """
                UNWIND $entities AS entity
                MERGE (e:HotpotEntity {entity_key: entity.entity_key})
                ON CREATE SET e.name = entity.name, e.created_at = timestamp()
                SET e.display_name = coalesce(e.display_name, entity.name),
                    e.name_norm = entity.name_norm,
                    e.graph_run_id = entity.graph_run_id,
                    e.run_id = entity.graph_run_id,
                    e.last_seen_at = timestamp()
                """,
                entities=list(entity_rows.values()),
            )
            edge_cypher = """
                UNWIND $edges AS edge
                MATCH (h:HotpotEntity {entity_key: edge.head_key})
                MATCH (t:HotpotEntity {entity_key: edge.tail_key})
                CREATE (h)-[r:HOTPOT_RELATION {relation_id: edge.relation_id}]->(t)
                SET r.type = edge.type,
                    r.type_norm = edge.type_norm,
                    r.head = edge.head,
                    r.tail = edge.tail,
                    r.graph_run_id = edge.graph_run_id,
                    r.run_id = edge.graph_run_id,
                    r.source_doc_id = edge.source_doc_id,
                    r.source_document_id = edge.source_document_id,
                    r.source_title = edge.source_title,
                    r.source_chunk_id = edge.source_chunk_id,
                    r.source_sentence_ids = edge.source_sentence_ids,
                    r.supporting_quote = edge.supporting_quote,
                    r.extractor = edge.extractor,
                    r.extraction_model = edge.extractor,
                    r.confidence = edge.confidence,
                    r.updated_at = timestamp()
                """
            for edge_slice in _slices(edges, 1000):
                session.run(edge_cypher, edges=edge_slice)
        session.run(
            """
            MERGE (g:GraphRun {graph_run_id:$gid})
            SET g.status = 'building',
                g.last_updated = datetime(),
                g.latest_chunk_count = coalesce(g.latest_chunk_count, 0) + $chunk_count,
                g.latest_relation_count = coalesce(g.latest_relation_count, 0) + $edge_count
            """,
            gid=graph_run_id,
            chunk_count=len(chunks),
            edge_count=len(edges),
        )


def build_hotpotqa_train_graph(
    graph_run_id: str = DEFAULT_GRAPH_RUN_ID,
    fulltext_index_name: str = DEFAULT_FULLTEXT_INDEX_NAME,
    artifact_dir: Path | None = None,
    force_rebuild: bool = False,
    max_docs: int | None = None,
    batch_size: int = 2000,
    max_edges_per_chunk: int = 2,
) -> dict[str, Any]:
    """Build or reuse a full HotpotQA train Neo4j graph."""
    uri, auth = neo4j_auth_from_env()
    artifact_dir = artifact_dir or Path("data/graph_runs") / graph_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "build_trace.jsonl"
    driver = GraphDatabase.driver(uri, auth=auth)
    started = time.perf_counter()
    try:
        driver.verify_connectivity()
        existing = graph_exists(driver, graph_run_id)
        if existing["exists"] and existing["status"] == "completed" and not force_rebuild:
            schema_status = ensure_schema(driver)
            schema_status.update(create_hotpot_fulltext_index(driver, fulltext_index_name))
            schema_status.update(create_hotpot_entity_fulltext_index(driver))
            manifest = {
                "graph_run_id": graph_run_id,
                "status": "reused",
                "neo4j_uri": uri,
                "fulltext_index_name": fulltext_index_name,
                "counts": existing,
                "artifact_dir": str(artifact_dir),
                "neo4j_graphrag": schema_status,
                "reuse_policy": "existing completed graph state reused; no rebuild",
            }
            write_json(artifact_dir / "run_manifest.json", manifest)
            return manifest
        schema_status: dict[str, Any] = {}
        if force_rebuild or existing["exists"]:
            schema_status.update(drop_hotpot_fulltext_index(driver, fulltext_index_name))
            delete_graph_run(driver, graph_run_id)
            if trace_path.exists():
                trace_path.unlink()
        schema_status.update(ensure_schema(driver))
        with driver.session() as session:
            session.run(
                """
                MERGE (g:GraphRun {graph_run_id:$gid})
                SET g.status='building',
                    g.dataset='hotpotqa',
                    g.build_split='train',
                    g.created_at=coalesce(g.created_at, datetime()),
                    g.builder='src.indexing.neo4j_hotpotqa_graph',
                    g.neo4j_graphrag_fulltext_index=$fts
                """,
                gid=graph_run_id,
                fts=fulltext_index_name,
            )

        chunk_batch: list[dict[str, Any]] = []
        edge_batch: list[dict[str, Any]] = []
        chunk_count = 0
        edge_count = 0
        for chunk in iter_unique_train_chunks(max_docs=max_docs):
            row = _chunk_row(chunk, graph_run_id)
            edges = _edge_rows(chunk, graph_run_id, max_edges_per_chunk=max_edges_per_chunk)
            chunk_batch.append(row)
            edge_batch.extend(edges)
            chunk_count += 1
            edge_count += len(edges)
            if len(chunk_batch) >= batch_size:
                _write_batch(driver, chunk_batch, edge_batch, graph_run_id)
                append_jsonl(
                    trace_path,
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "chunks_written": chunk_count,
                        "edges_written": edge_count,
                        "elapsed_sec": time.perf_counter() - started,
                    },
                )
                if chunk_count % max(batch_size * 10, 1) == 0:
                    print(
                        f"hotpotqa train graph progress: chunks={chunk_count} edges={edge_count} elapsed_sec={time.perf_counter() - started:.1f}",
                        flush=True,
                    )
                chunk_batch = []
                edge_batch = []
        if chunk_batch or edge_batch:
            _write_batch(driver, chunk_batch, edge_batch, graph_run_id)

        counts = graph_exists(driver, graph_run_id)
        with driver.session() as session:
            session.run(
                """
                MATCH (g:GraphRun {graph_run_id:$gid})
                SET g.status='completed',
                    g.completed_at=datetime(),
                    g.chunk_count=$chunks,
                    g.entity_count=$entities,
                    g.relation_count=$relations,
                    g.max_docs=$max_docs,
                    g.max_edges_per_chunk=$max_edges_per_chunk
                """,
                gid=graph_run_id,
                chunks=counts["chunks"],
                entities=counts["entities"],
                relations=counts["relations"],
                max_docs=max_docs,
                max_edges_per_chunk=max_edges_per_chunk,
            )
        counts = graph_exists(driver, graph_run_id)
        counts["status"] = "completed"
        schema_status.update(create_hotpot_fulltext_index(driver, fulltext_index_name))
        schema_status.update(create_hotpot_entity_fulltext_index(driver))
        manifest = {
            "graph_run_id": graph_run_id,
            "status": "completed",
            "neo4j_uri": uri,
            "fulltext_index_name": fulltext_index_name,
            "artifact_dir": str(artifact_dir),
            "build_split": "train",
            "eval_split": "validation",
            "dataset_source": "huggingface:hotpot_qa/distractor",
            "public_test_split_available": False,
            "chunking": {"max_sentences_per_chunk": 3, "overlap_sentences": 1},
            "graph_extractor": "deterministic_titlecase_cooccurrence_v2",
            "qa_model": "deterministic_evidence_sentence_v1 unless external generator configured",
            "graph_create_model": "deterministic_titlecase_cooccurrence_v2; config.py Ollama graph_create_model observed separately",
            "counts": counts,
            "neo4j_graphrag": schema_status,
            "elapsed_sec": time.perf_counter() - started,
            "reuse_policy": "safe to reuse for all ablations by graph_run_id",
        }
        write_json(artifact_dir / "run_manifest.json", manifest)
        return manifest
    finally:
        driver.close()


def lucene_escape(query: str) -> str:
    """Escape a query for Neo4j fulltext lookup while keeping useful terms."""
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", str(query or ""))
    return " ".join(terms[:32]) or "hotpotqa"


def build_hotpotqa_graph_from_chunks(
    chunks: list[dict[str, Any]],
    graph_run_id: str,
    *,
    fulltext_index_name: str = DEFAULT_FULLTEXT_INDEX_NAME,
    artifact_dir: Path | None = None,
    force_rebuild: bool = False,
    batch_size: int = 2000,
    max_edges_per_chunk: int = 2,
    extractor_mode: str = "deterministic",
    use_llm_extraction: bool = False,
    llm_model: str | None = None,
    llm_host: str | None = None,
    llm_chunk_limit: int | None = None,
    extraction_temperature: float = 0.0,
) -> dict[str, Any]:
    """Build or reuse a HotpotQA graph from an explicit chunk list."""
    uri, auth = neo4j_auth_from_env()
    artifact_dir = artifact_dir or Path("data/graph_runs") / graph_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifact_dir / "build_trace.jsonl"
    driver = GraphDatabase.driver(uri, auth=auth)
    started = time.perf_counter()
    client = None
    if use_llm_extraction:
        from ollama import Client

        client = Client(host=llm_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    try:
        driver.verify_connectivity()
        existing = graph_exists(driver, graph_run_id)
        if existing["exists"] and existing["status"] == "completed" and not force_rebuild:
            schema_status = ensure_schema(driver)
            schema_status.update(create_hotpot_fulltext_index(driver, fulltext_index_name))
            schema_status.update(create_hotpot_entity_fulltext_index(driver))
            manifest = {
                "graph_run_id": graph_run_id,
                "status": "reused",
                "neo4j_uri": uri,
                "fulltext_index_name": fulltext_index_name,
                "counts": existing,
                "artifact_dir": str(artifact_dir),
                "neo4j_graphrag": schema_status,
                "reuse_policy": "existing completed graph state reused; no rebuild",
            }
            write_json(artifact_dir / "run_manifest.json", manifest)
            return manifest
        schema_status: dict[str, Any] = {}
        if force_rebuild or existing["exists"]:
            schema_status.update(drop_hotpot_fulltext_index(driver, fulltext_index_name))
            delete_graph_run(driver, graph_run_id)
            if trace_path.exists():
                trace_path.unlink()
        schema_status.update(ensure_schema(driver))
        with driver.session() as session:
            session.run(
                """
                MERGE (g:GraphRun {graph_run_id:$gid})
                SET g.status='building',
                    g.dataset='hotpotqa',
                    g.build_split='train_sample',
                    g.created_at=coalesce(g.created_at, datetime()),
                    g.builder='src.indexing.neo4j_hotpotqa_graph.build_hotpotqa_graph_from_chunks',
                    g.neo4j_graphrag_fulltext_index=$fts
                """,
                gid=graph_run_id,
                fts=fulltext_index_name,
            )
        chunk_batch: list[dict[str, Any]] = []
        edge_batch: list[dict[str, Any]] = []
        chunk_count = 0
        edge_count = 0
        for idx, chunk in enumerate(chunks, start=1):
            chunk_batch.append(_chunk_row(chunk, graph_run_id))
            det_edges = extract_relations_from_chunk(chunk, max_entities=3, max_edges=max_edges_per_chunk)
            llm_edges: list[dict[str, Any]] = []
            if use_llm_extraction and client and llm_model and (llm_chunk_limit is None or idx <= llm_chunk_limit):
                try:
                    llm_edges = extract_relations_from_chunk_ollama(
                        chunk,
                        client,
                        llm_model,
                        temperature=extraction_temperature,
                        max_edges=max_edges_per_chunk,
                    )
                except Exception as exc:
                    append_jsonl(
                        trace_path,
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "chunk_id": chunk.get("chunk_id"),
                            "stage": "llm_relation_extraction",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
            if extractor_mode == "deterministic":
                selected_edges = det_edges
            elif extractor_mode == "llm":
                selected_edges = llm_edges
            else:
                selected_edges = merge_relation_edges(det_edges, llm_edges)
            for edge in selected_edges:
                head_key = entity_key(edge["head"])
                tail_key = entity_key(edge["tail"])
                if not head_key or not tail_key or head_key == tail_key:
                    continue
                relation = str(edge.get("relation") or "co_occurs_with")
                relation_id = f"{graph_run_id}::{head_key}::{relation}::{tail_key}::{edge['source_chunk_id']}::{len(edge_batch)}"
                edge_batch.append(
                    {
                        "relation_id": relation_id,
                        "head": edge["head"],
                        "head_norm": head_key,
                        "head_key": f"{graph_run_id}::{head_key}",
                        "tail": edge["tail"],
                        "tail_norm": tail_key,
                        "tail_key": f"{graph_run_id}::{tail_key}",
                        "type": relation,
                        "type_norm": relation,
                        "source_doc_id": edge.get("source_doc_id"),
                        "source_document_id": edge.get("source_doc_id"),
                        "source_title": edge.get("source_title"),
                        "source_chunk_id": edge.get("source_chunk_id"),
                        "source_sentence_ids": list(edge.get("source_sentence_ids", [])),
                        "supporting_quote": edge.get("supporting_quote", ""),
                        "extractor": edge.get("extractor", "rule_based_titlecase_comention_v1"),
                        "confidence": float(edge.get("confidence", 0.6) or 0.6),
                        "graph_run_id": graph_run_id,
                        "run_id": graph_run_id,
                    }
                )
            chunk_count += 1
            if len(chunk_batch) >= batch_size:
                _write_batch(driver, chunk_batch, edge_batch, graph_run_id)
                append_jsonl(
                    trace_path,
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "chunks_written": chunk_count,
                        "edges_written": edge_count + len(edge_batch),
                        "elapsed_sec": time.perf_counter() - started,
                    },
                )
                chunk_batch = []
                edge_count += len(edge_batch)
                edge_batch = []
        if chunk_batch or edge_batch:
            _write_batch(driver, chunk_batch, edge_batch, graph_run_id)
            edge_count += len(edge_batch)
        counts = graph_exists(driver, graph_run_id)
        with driver.session() as session:
            session.run(
                """
                MATCH (g:GraphRun {graph_run_id:$gid})
                SET g.status='completed',
                    g.completed_at=datetime(),
                    g.chunk_count=$chunks,
                    g.entity_count=$entities,
                    g.relation_count=$relations
                """,
                gid=graph_run_id,
                chunks=counts["chunks"],
                entities=counts["entities"],
                relations=counts["relations"],
            )
        counts = graph_exists(driver, graph_run_id)
        counts["status"] = "completed"
        schema_status.update(create_hotpot_fulltext_index(driver, fulltext_index_name))
        schema_status.update(create_hotpot_entity_fulltext_index(driver))
        manifest = {
            "graph_run_id": graph_run_id,
            "status": "completed",
            "neo4j_uri": uri,
            "fulltext_index_name": fulltext_index_name,
            "artifact_dir": str(artifact_dir),
            "build_split": "train_sample",
            "dataset_source": "processed_train_sample_chunks",
            "chunking": {"max_sentences_per_chunk": 3, "overlap_sentences": 1},
            "graph_extractor": extractor_mode,
            "use_llm_extraction": use_llm_extraction,
            "llm_model": llm_model,
            "counts": counts,
            "neo4j_graphrag": schema_status,
            "elapsed_sec": time.perf_counter() - started,
            "reuse_policy": "safe to reuse for all ablations by graph_run_id",
        }
        write_json(artifact_dir / "run_manifest.json", manifest)
        write_jsonl(artifact_dir / "chunks.jsonl", chunks)
        return manifest
    finally:
        driver.close()
