# Graph Schema

The Neo4j knowledge graph models the GDPR's structure and the cross-references
between articles. It is built from `data/docs/gdpr_articles.json` by
[src/graph/load_graph.py](../src/graph/load_graph.py).

## Nodes

| Label | Key | Properties |
|-------|-----|------------|
| `Article` | `number` (unique) | `number`, `title`, `full_text`, `url` |
| `Chapter` | `name` (unique) | `name` |
| `Recital` | `number` (unique) | `number` |

## Relationships

| Relationship | Meaning |
|--------------|---------|
| `(:Article)-[:IN_CHAPTER]->(:Chapter)` | the article belongs to a GDPR chapter |
| `(:Article)-[:HAS_RECITAL]->(:Recital)` | a recital that explains the article |
| `(:Article)-[:REFERENCES]->(:Article)` | the article's text cites another article |

```
(:Chapter) ◀─[:IN_CHAPTER]─ (:Article) ─[:HAS_RECITAL]─▶ (:Recital)
                                │  ▲
                                └──┘ [:REFERENCES]   (article → article)
```

## Constraints

Defined in [src/graph/schema.py](../src/graph/schema.py); created idempotently by
the loader. Uniqueness constraints also back the lookups used by `MERGE`:

- `Article.number` unique
- `Chapter.name` unique
- `Recital.number` unique

## Cross-reference extraction

`REFERENCES` edges come from parsing each article's `full_text` with
[src/graph/extract_entities.py](../src/graph/extract_entities.py):

- Matches `Article N`, lists (`Articles 13, 14 and 15`) and ranges
  (`Articles 44 to 49`, expanded inclusively).
- Ignores paragraph references — `Article 6(1)` yields article **6** only.
- Drops self-references and anything outside articles 1–99.

## Loaded scale

A full load produces roughly:

| Element | Count |
|---------|-------|
| `Article` | 99 |
| `Chapter` | 11 |
| `Recital` | 172 |
| `IN_CHAPTER` | 99 |
| `HAS_RECITAL` | 227 |
| `REFERENCES` | 314 |

## Example queries

Articles that reference Article 6 (the lawful-basis article):
```cypher
MATCH (a:Article)-[:REFERENCES]->(:Article {number: 6})
RETURN a.number, a.title ORDER BY a.number;
```

Neighbours of an article (references in/out + chapter siblings) — the traversal
used by the graph retriever:
```cypher
MATCH (a:Article {number: 6})
CALL (a) {
    MATCH (a)-[:REFERENCES]->(b:Article)                          RETURN b
    UNION
    MATCH (a)<-[:REFERENCES]-(b:Article)                          RETURN b
    UNION
    MATCH (a)-[:IN_CHAPTER]->(:Chapter)<-[:IN_CHAPTER]-(b:Article) RETURN b
}
RETURN DISTINCT b.number, b.title ORDER BY b.number;
```
