from typing import Dict

from neo4j import GraphDatabase


class Neo4jConnector:
    def __init__(self, uri: str, auth: tuple):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.uri = uri

    def close(self):
        if self.driver:
            self.driver.close()

    def get_driver(self):
        return self.driver

    def verify_connectivity(self):
        self.driver.verify_connectivity()
        print(f"Neo4j connected: {self.uri}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def clean_database(driver, dataset_id: str, clean_all: bool = False) -> Dict[str, int]:
    with driver.session() as session:
        if clean_all:
            print("Cleaning entire Neo4j database...")
            deleted_relations = session.run("MATCH ()-[r]->() DELETE r RETURN count(r) AS cnt").single()["cnt"]
            deleted_nodes = session.run("MATCH (n) DELETE n RETURN count(n) AS cnt").single()["cnt"]
            print(f"  Deleted {deleted_nodes} nodes and {deleted_relations} relationships")
            return {
                "deleted_chunks": deleted_nodes,
                "deleted_mentions": deleted_relations,
                "deleted_entities": 0,
                "deleted_relations": 0,
            }

        print(f"Cleaning dataset-specific graph data for dataset_id={dataset_id!r} ...")
        deleted_mentions = session.run(
            """
            MATCH (c:Chunk {dataset: $dataset})-[m:MENTIONS]->(:Entity)
            DELETE m
            RETURN count(m) AS cnt
            """,
            dataset=dataset_id,
        ).single()["cnt"]

        deleted_relations = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
            MATCH (e)-[r:RELATION]-()
            DELETE r
            RETURN count(r) AS cnt
            """
        ).single()["cnt"]

        deleted_entities = session.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
              AND NOT (e)-[:RELATION]-()
            DELETE e
            RETURN count(e) AS cnt
            """
        ).single()["cnt"]

        deleted_chunks = session.run(
            """
            MATCH (c:Chunk {dataset: $dataset})
            DELETE c
            RETURN count(c) AS cnt
            """,
            dataset=dataset_id,
        ).single()["cnt"]

        print(f"  Deleted {deleted_chunks} chunks")
        print(f"  Deleted {deleted_mentions} mentions")
        print(f"  Deleted {deleted_entities} entities")
        print(f"  Deleted {deleted_relations} relations")
        return {
            "deleted_chunks": deleted_chunks,
            "deleted_mentions": deleted_mentions,
            "deleted_entities": deleted_entities,
            "deleted_relations": deleted_relations,
        }


def ensure_entity_index(driver) -> None:
    with driver.session() as session:
        try:
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            print("  Entity name unique constraint ready")
        except Exception:
            session.run(
                "CREATE INDEX entity_name_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name)"
            )
            print("  Entity name index ready")

        try:
            session.run(
                "CREATE CONSTRAINT entity_name_norm_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name_norm IS UNIQUE"
            )
            print("  Entity name_norm unique constraint ready")
        except Exception:
            session.run(
                "CREATE INDEX entity_name_norm_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.name_norm)"
            )
            print("  Entity name_norm index ready")


def ensure_vector_index(
    driver,
    name: str,
    label: str,
    prop: str,
    dimensions: int,
    similarity: str = "cosine",
) -> None:
    with driver.session() as session:
        existing = session.run("SHOW INDEXES").data()
        if any(idx.get("name") == name for idx in existing):
            print(f"  Vector index '{name}' already exists")
            return

        cypher = f"""
        CREATE VECTOR INDEX {name}
        FOR (n:{label}) ON (n.{prop})
        OPTIONS {{ indexConfig: {{ `vector.dimensions`: {dimensions}, `vector.similarity_function`: '{similarity}' }} }}
        """
        session.run(cypher)
        session.run("CALL db.awaitIndexes()")
        print(f"  Vector index '{name}' created")


def ensure_fulltext_index(driver, name: str, label: str, prop: str = "text") -> bool:
    with driver.session() as session:
        existing = session.run("SHOW INDEXES").data()
        if any(idx.get("name") == name for idx in existing):
            print(f"  Fulltext index '{name}' already exists")
            return True

        try:
            session.run(f"CREATE FULLTEXT INDEX {name} FOR (n:{label}) ON EACH [n.{prop}]")
            session.run("CALL db.awaitIndexes()")
            print(f"  Fulltext index '{name}' created")
            return True
        except Exception as exc:
            print(f"  Failed to create fulltext index '{name}': {exc}")
            return False


def ensure_graph_run_index(driver) -> None:
    with driver.session() as session:
        try:
            session.run(
                "CREATE CONSTRAINT graph_run_unique IF NOT EXISTS "
                "FOR (g:GraphRun) REQUIRE g.graph_run_id IS UNIQUE"
            )
            print("  GraphRun unique constraint ready")
        except Exception as exc:
            print(f"  Warning: failed to ensure GraphRun constraint: {exc}")
