"""Graph expansion and path enumeration."""

from __future__ import annotations

from typing import Any

from src.graph.canonicalization import entity_key


def expand_graph(store: Any, seed_entities: list[str], depth: int = 1, max_nodes_per_hop: int = 10) -> dict[str, Any]:
    """Expand a local graph around linked seed entities."""
    graph = store.graph
    seed_keys = [entity_key(seed) for seed in seed_entities if entity_key(seed) in graph]
    visited = set(seed_keys)
    frontier = list(seed_keys)
    edges: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    for hop in range(depth):
        next_frontier: list[str] = []
        for node in frontier[:max_nodes_per_hop]:
            neighbors = list(graph.successors(node))[:max_nodes_per_hop] + list(graph.predecessors(node))[:max_nodes_per_hop]
            for nb in neighbors[:max_nodes_per_hop]:
                edge_datas = []
                if graph.has_edge(node, nb):
                    edge_datas.extend(graph.get_edge_data(node, nb).values())
                if graph.has_edge(nb, node):
                    edge_datas.extend(graph.get_edge_data(nb, node).values())
                for data in edge_datas[:2]:
                    if not data.get("source_chunk_id"):
                        continue
                    record = dict(data)
                    record["head"] = data.get("head")
                    record["tail"] = data.get("tail")
                    edges.append(record)
                    nodes = [graph.nodes[node].get("label", node), graph.nodes[nb].get("label", nb)]
                    paths.append(
                        {
                            "path_id": f"path_{len(paths):05d}",
                            "nodes": nodes,
                            "edges": [
                                {
                                    "head": record.get("head"),
                                    "relation": record.get("relation"),
                                    "tail": record.get("tail"),
                                    "source_chunk_id": record.get("source_chunk_id"),
                                    "source_sentence_ids": record.get("source_sentence_ids", []),
                                }
                            ],
                            "path_score": 0.0,
                        }
                    )
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier[:max_nodes_per_hop]
    if depth >= 2:
        two_hop_paths: list[dict[str, Any]] = []
        for seed in seed_keys:
            for mid in list(graph.successors(seed))[:max_nodes_per_hop]:
                for end in list(graph.successors(mid))[:max_nodes_per_hop]:
                    if end == seed:
                        continue
                    data1 = next(iter(graph.get_edge_data(seed, mid).values()))
                    data2 = next(iter(graph.get_edge_data(mid, end).values()))
                    if not data1.get("source_chunk_id") or not data2.get("source_chunk_id"):
                        continue
                    two_hop_paths.append(
                        {
                            "path_id": f"path_{len(paths) + len(two_hop_paths):05d}",
                            "nodes": [
                                graph.nodes[seed].get("label", seed),
                                graph.nodes[mid].get("label", mid),
                                graph.nodes[end].get("label", end),
                            ],
                            "edges": [
                                {
                                    "head": data1.get("head"),
                                    "relation": data1.get("relation"),
                                    "tail": data1.get("tail"),
                                    "source_chunk_id": data1.get("source_chunk_id"),
                                    "source_sentence_ids": data1.get("source_sentence_ids", []),
                                },
                                {
                                    "head": data2.get("head"),
                                    "relation": data2.get("relation"),
                                    "tail": data2.get("tail"),
                                    "source_chunk_id": data2.get("source_chunk_id"),
                                    "source_sentence_ids": data2.get("source_sentence_ids", []),
                                },
                            ],
                            "path_score": 0.0,
                        }
                    )
        paths.extend(two_hop_paths)
    entity_labels = [graph.nodes[n].get("label", n) for n in visited]
    return {"seed_keys": seed_keys, "entities": entity_labels, "edges": edges, "paths": paths}
