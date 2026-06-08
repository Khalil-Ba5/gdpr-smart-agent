"""Neo4j graph schema for the GDPR knowledge graph.

Model
-----
    (:Article {number, title, full_text, url})
    (:Chapter {name})
    (:Recital {number})

    (:Article)-[:IN_CHAPTER]->(:Chapter)
    (:Article)-[:HAS_RECITAL]->(:Recital)
    (:Article)-[:REFERENCES]->(:Article)   # cross-refs parsed from article text

This module owns the driver factory and the uniqueness constraints so the
loader and the retriever share one definition.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

load_dotenv()

# Uniqueness constraints (also create the backing indexes used by MERGE/lookups).
CONSTRAINTS = [
    "CREATE CONSTRAINT article_number IF NOT EXISTS "
    "FOR (a:Article) REQUIRE a.number IS UNIQUE",
    "CREATE CONSTRAINT chapter_name IF NOT EXISTS "
    "FOR (c:Chapter) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT recital_number IF NOT EXISTS "
    "FOR (r:Recital) REQUIRE r.number IS UNIQUE",
]


def get_driver() -> Driver:
    """Create a Neo4j driver from NEO4J_* environment variables."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def apply_constraints(driver: Driver) -> None:
    """Create the uniqueness constraints (idempotent)."""
    with driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
