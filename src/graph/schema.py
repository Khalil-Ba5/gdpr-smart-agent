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
from functools import lru_cache

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
    """Create a *new* Neo4j driver from NEO4J_* environment variables.

    Callers own the returned driver's lifecycle and must ``close()`` it (see
    ``load_graph.py`` / ``check_setup.py``). For the retrieval hot path, use
    ``get_cached_driver()`` instead — a shared driver that is never closed.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


@lru_cache(maxsize=1)
def get_cached_driver() -> Driver:
    """Process-wide singleton driver for the retrieval hot path.

    ``neo4j.Driver`` instances are meant to be long-lived and manage their own
    connection pool internally, so opening/closing one per request (as the API
    and MCP server do) is needless overhead. Never call ``.close()`` on this —
    it's shared for the life of the process.
    """
    return get_driver()


def apply_constraints(driver: Driver) -> None:
    """Create the uniqueness constraints (idempotent)."""
    with driver.session() as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)
