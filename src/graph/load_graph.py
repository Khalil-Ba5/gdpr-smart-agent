"""Load the GDPR knowledge graph into Neo4j from the scraped JSON.

Idempotent: every write uses MERGE, so re-running updates in place rather
than duplicating nodes/edges.

Usage:
    python -m src.graph.load_graph
"""

from __future__ import annotations

import json
from pathlib import Path

from neo4j import Driver

from src.graph.extract_entities import extract_article_references
from src.graph.schema import apply_constraints, get_driver

JSON_PATH = "data/docs/gdpr_articles.json"

# One article + its chapter and recital links. Run per article.
_UPSERT_ARTICLE = """
MERGE (a:Article {number: $number})
SET a.title = $title, a.full_text = $full_text, a.url = $url
WITH a
CALL (a) {
    WITH a WHERE $chapter <> ''
    MERGE (c:Chapter {name: $chapter})
    MERGE (a)-[:IN_CHAPTER]->(c)
}
WITH a
UNWIND $recitals AS recital_number
    MERGE (r:Recital {number: recital_number})
    MERGE (a)-[:HAS_RECITAL]->(r)
"""

# Cross-reference edges, run after all articles exist so both ends are present.
_UPSERT_REFERENCES = """
MATCH (a:Article {number: $number})
UNWIND $refs AS ref
    MATCH (b:Article {number: ref})
    MERGE (a)-[:REFERENCES]->(b)
"""


def load_articles(driver: Driver, articles: list[dict]) -> dict[str, int]:
    """Write articles, chapters, recitals and cross-references. Returns counts."""
    apply_constraints(driver)

    with driver.session() as session:
        # Pass 1: nodes + chapter/recital edges.
        for art in articles:
            session.run(
                _UPSERT_ARTICLE,
                number=art["article_number"],
                title=art.get("title", ""),
                full_text=art.get("full_text", ""),
                url=art.get("url", ""),
                chapter=art.get("chapter", "") or "",
                recitals=[int(r) for r in art.get("recitals", [])],
            )

        # Pass 2: REFERENCES edges (needs every article node to exist first).
        ref_edges = 0
        for art in articles:
            refs = extract_article_references(
                art.get("full_text", ""), source_article=art["article_number"]
            )
            if refs:
                session.run(_UPSERT_REFERENCES, number=art["article_number"], refs=refs)
                ref_edges += len(refs)

    return {"articles": len(articles), "reference_edges": ref_edges}


def main(json_path: str = JSON_PATH) -> None:
    articles = json.loads(Path(json_path).read_text(encoding="utf-8"))
    driver = get_driver()
    try:
        counts = load_articles(driver, articles)
    finally:
        driver.close()
    print(f"Loaded {counts['articles']} articles, {counts['reference_edges']} REFERENCES edges.")


if __name__ == "__main__":
    main()
