"""Graph retrieval over the Neo4j GDPR knowledge graph.

Answers *relationship* questions that vector search handles poorly:
"what articles reference Article 6?", "what else is in the same chapter?".

Given one or more anchor article numbers (usually parsed from the user's
question), it returns neighbouring articles via the REFERENCES and IN_CHAPTER
relationships, packaged as the same ``RetrievedChunk`` type the vector
retriever uses so the generate node can treat both sources uniformly.
"""

from __future__ import annotations

import re

from src.graph.schema import get_cached_driver
from src.rag.vector_retriever import RetrievedChunk

# "Article 6", "Art. 17", "article 6(1)" -> 6 / 17 / 6
_ARTICLE_IN_QUERY = re.compile(r"\bart(?:icle)?\.?\s*(\d+)", re.IGNORECASE)

# Outgoing + incoming references and chapter siblings of the anchor articles.
_NEIGHBOURS = """
UNWIND $numbers AS num
MATCH (a:Article {number: num})
CALL (a) {
    MATCH (a)-[:REFERENCES]->(b:Article)            RETURN b
    UNION
    MATCH (a)<-[:REFERENCES]-(b:Article)            RETURN b
    UNION
    MATCH (a)-[:IN_CHAPTER]->(:Chapter)<-[:IN_CHAPTER]-(b:Article) RETURN b
}
WITH DISTINCT b
WHERE NOT b.number IN $numbers
RETURN b.number AS number, b.title AS title, b.full_text AS full_text, b.url AS url
ORDER BY number
LIMIT $limit
"""


def parse_article_numbers(text: str) -> list[int]:
    """Pull explicit article numbers out of a natural-language question."""
    nums = {int(m.group(1)) for m in _ARTICLE_IN_QUERY.finditer(text or "")}
    return sorted(n for n in nums if 1 <= n <= 99)


def _record_to_chunk(rec: dict) -> RetrievedChunk:
    return RetrievedChunk(
        text=rec["full_text"] or "",
        article_number=rec["number"],
        paragraph_number="",  # whole-article node, no single paragraph
        title=rec["title"] or "",
        chapter="",
        url=rec["url"] or "",
    )


def retrieve_related(numbers: list[int], limit: int = 6, driver=None) -> list[RetrievedChunk]:
    """Return articles related to the given anchor ``numbers``.

    Uses the shared, process-wide cached driver by default (see
    ``src.graph.schema.get_cached_driver``) — pass ``driver`` explicitly (as
    the tests do) to use a different one instead.
    """
    if not numbers:
        return []

    driver = driver or get_cached_driver()
    with driver.session() as session:
        records = session.run(_NEIGHBOURS, numbers=numbers, limit=limit).data()

    return [_record_to_chunk(rec) for rec in records]


def retrieve(query: str, limit: int = 6, driver=None) -> list[RetrievedChunk]:
    """Convenience wrapper: parse article numbers from ``query`` then expand."""
    return retrieve_related(parse_article_numbers(query), limit=limit, driver=driver)


if __name__ == "__main__":
    for chunk in retrieve("What articles relate to Article 6?"):
        print(f"[{chunk.citation}] {chunk.title}")
