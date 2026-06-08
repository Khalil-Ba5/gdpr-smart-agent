# GDPR Smart Agent

A question-answering agent over the EU **GDPR** that combines **two retrieval
strategies** and cites the articles it relies on:

- **Vector RAG** — semantic search over GDPR paragraphs (Chroma + OpenAI embeddings)
  for definitional / substantive questions ("*what is personal data?*").
- **Graph RAG** — traversal of a Neo4j knowledge graph of article cross-references
  and chapters for relationship questions ("*which articles reference Article 6?*").

A **LangGraph** agent routes each question to the right retriever and generates a
grounded, cited answer. The agent is exposed over a **FastAPI** HTTP API and an
**MCP server**.

```
                      ┌─────────────┐
question ──▶ route ──▶│  vector_rag │──▶ generate ──▶ cited answer
                  └──▶│  graph_rag  │──▶
                      └─────────────┘
```

## Architecture

| Layer | Module | Notes |
|-------|--------|-------|
| Scrape | [scripts/scrape_gdpr.py](scripts/scrape_gdpr.py) | 99 articles from gdpr-info.eu → JSON |
| Vector ingest | [src/rag/ingest.py](src/rag/ingest.py) | paragraph chunks → Chroma (`text-embedding-3-small`) |
| Vector retrieval | [src/rag/vector_retriever.py](src/rag/vector_retriever.py) | similarity search → cited chunks |
| Graph build | [src/graph/](src/graph/) | schema, cross-ref extraction, Neo4j loader |
| Graph retrieval | [src/rag/graph_retriever.py](src/rag/graph_retriever.py) | Cypher traversal → cited chunks |
| Agent | [src/agent/](src/agent/) | LangGraph: route → retrieve → generate |
| API | [src/api.py](src/api.py) | FastAPI `/ask`, `/health` |
| MCP | [src/mcp/server.py](src/mcp/server.py) | `search_gdpr`, `related_articles`, `ask_gdpr` |

See [docs/architecture.md](docs/architecture.md) and [docs/graph-schema.md](docs/graph-schema.md).

## Setup

Requires **Python 3.11**, an **OpenAI API key**, and **Docker** (for Neo4j).

```bash
# 1. Environment
python -m venv venv
venv\Scripts\activate            # Windows ( source venv/bin/activate on macOS/Linux )
pip install -r requirements.txt

# 2. Secrets — copy the template and fill in your keys
copy .env.example .env           # cp on macOS/Linux

# 3. Start Neo4j
docker compose up -d neo4j

# 4. Build the indexes  (or run both at once: python -m scripts.seed_data)
python -m src.rag.ingest         # vector index → chroma_db/
python -m src.graph.load_graph   # knowledge graph → Neo4j

# 5. Sanity-check connectivity (LLM, Chroma, Neo4j)
python -m scripts.check_setup
```

## Usage

**CLI** (one-off question):
```bash
python -m src.agent.graph "Which articles relate to Article 6?"
```

**Interactive chat** (REPL):
```bash
python -m src.app
```

**HTTP API**:
```bash
uvicorn src.api:app --reload
# POST http://127.0.0.1:8000/ask   {"question": "...", "k": 4}
# docs at http://127.0.0.1:8000/docs
```

**MCP server** (stdio) — add to an MCP client (e.g. Claude Desktop):
```json
{
  "mcpServers": {
    "gdpr": {
      "command": "<abs path>/venv/Scripts/python.exe",
      "args": ["-m", "src.mcp.server"],
      "cwd": "<abs path to gdpr-smart-agent>"
    }
  }
}
```

## Run the whole stack with Docker

```bash
docker compose up --build        # Neo4j + API together
```
The API container reaches Neo4j over the compose network. Build the vector index
(`chroma_db/`) and load the graph first (steps 4 above) — the index is mounted
read-only into the container.

## Configuration (`.env`)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | embeddings + default LLM |
| `LLM_PROVIDER` | `openai` (default) or `anthropic` |
| `ANTHROPIC_API_KEY` | required if `LLM_PROVIDER=anthropic` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | graph connection |

## Tests & evaluation

```bash
python -m pytest            # unit tests — network-free (LLM, Chroma, Neo4j mocked)
python -m scripts.evaluate  # retrieval quality + router accuracy (embeddings only)
python -m scripts.judge     # LLM-as-judge faithfulness on a sample (calls the LLM)
```
See [docs/evaluation-results.md](docs/evaluation-results.md) for current scores.
