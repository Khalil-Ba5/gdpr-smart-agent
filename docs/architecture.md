# Architecture

## Overview

The GDPR Smart Agent answers natural-language questions about the GDPR with
citations, using a **dual-retrieval** design orchestrated by a LangGraph agent.

```
                          ┌──────────────────┐
                          │      route       │  classify_route(question)
                          └────────┬─────────┘
            "vector"  ┌────────────┴────────────┐  "graph"
                      ▼                          ▼
            ┌───────────────────┐     ┌────────────────────┐
            │  vector_retrieve  │     │   graph_retrieve   │
            │  (Chroma + OpenAI)│     │  (Neo4j / Cypher)  │
            └─────────┬─────────┘     └─────────┬──────────┘
                      └────────────┬────────────┘
                                   ▼
                          ┌──────────────────┐
                          │     generate     │  LLM, grounded + cited
                          └──────────────────┘
```

Both retrievers return the same `RetrievedChunk` type, so `generate` is
source-agnostic and the answer format is identical regardless of route.

## Routing

`classify_route` (in [src/agent/nodes.py](../src/agent/nodes.py)) is a
lightweight, deterministic heuristic — no extra LLM call:

- **graph** — the question names a specific article (`Article 6`, `Art. 17`)
  **and** contains a relationship cue (*relate, reference, connected, …*).
- **vector** — everything else (definitions, substantive questions).

The graph branch falls back to vector retrieval if the graph returns nothing.

## Components

- **Ingestion** ([src/rag/ingest.py](../src/rag/ingest.py)) — one chunk per GDPR
  paragraph, embedded with `text-embedding-3-small`, persisted to Chroma. Rich
  metadata (article, paragraph, title, chapter, url) drives citations.
- **Vector retrieval** ([src/rag/vector_retriever.py](../src/rag/vector_retriever.py))
  — `similarity_search_with_score` → `RetrievedChunk` with a `citation` property.
- **Graph build** ([src/graph/](../src/graph/)) — schema + constraints, regex
  cross-reference extraction, idempotent MERGE loader. See
  [graph-schema.md](graph-schema.md).
- **Graph retrieval** ([src/rag/graph_retriever.py](../src/rag/graph_retriever.py))
  — parses article numbers from the question, expands via `REFERENCES`
  (both directions) and chapter siblings.
- **Agent** ([src/agent/](../src/agent/)) — `state.py` (typed state),
  `nodes.py` (route / retrieve / generate), `graph.py` (compiled StateGraph,
  cached). Public entry point: `answer_question(question, k)`.

## Interfaces

- **CLI** — `python -m src.agent.graph "<question>"`.
- **HTTP** — FastAPI ([src/api.py](../src/api.py)): `POST /ask`, `GET /health`.
  The graph is compiled once at startup via the lifespan handler.
- **MCP** — stdio server ([src/mcp/server.py](../src/mcp/server.py)) exposing
  `search_gdpr`, `related_articles`, `ask_gdpr`.

## Model configuration

Default LLM is OpenAI `gpt-4o`. Set `LLM_PROVIDER=anthropic` (with
`ANTHROPIC_API_KEY`) to use Claude instead — see `_get_llm` in
[src/agent/nodes.py](../src/agent/nodes.py). Embeddings always use OpenAI, since
the Chroma index was built with `text-embedding-3-small`.

## Testing

The suite ([tests/](../tests/)) is fully network-free: the LLM, Chroma store and
Neo4j driver are mocked, so `pytest` runs fast and without API keys or a running
database. Live behaviour is exercised manually via the CLI / API.
