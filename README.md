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
| MCP | [src/mcp/server.py](src/mcp/server.py) | `search_gdpr`, `related_articles`, `ask_gdpr`, `ask_gdpr_client` |
| Frontend | [src/streamlit_app.py](src/streamlit_app.py) | Streamlit chat UI over the API |

See [docs/architecture.md](docs/architecture.md) and [docs/graph-schema.md](docs/graph-schema.md).

## How it works

There are two separate flows: a one-time **offline** flow that builds the two
indexes, and the **runtime** flow that answers a question. Every interface
(CLI, API, MCP, Streamlit) calls the same function — `answer_question()` in
[src/agent/graph.py](src/agent/graph.py) — so the flow below is identical no
matter which one you use.

```
question
   │
   ▼
┌─────────────────────┐
│ route_node            │  classify_route(question):
│                        │    names an article AND uses a relational word
│                        │    ("relate", "reference", "connected", ...)
│                        │        → "graph"
│                        │    otherwise → "vector"
└───────────┬────────────┘
            │
    ┌───────┴────────┐
    ▼ "vector"         ▼ "graph"
┌──────────────┐   ┌───────────────────────┐
│ retrieve_node  │   │ graph_retrieve_node     │
│                │   │                         │
│ Chroma         │   │ Neo4j Cypher: articles  │
│ similarity     │   │ that REFERENCE (in/out) │
│ search, top-k  │   │ or share a Chapter with │
│ paragraph      │   │ the article named in    │
│ chunks         │   │ the question            │
│                │   │                         │
│                │   │ Falls back to           │
│                │   │ retrieve_node's vector   │
│                │   │ search if the graph      │
│                │   │ returns nothing, or if   │
│                │   │ Neo4j itself is          │
│                │   │ unreachable              │
└───────┬────────┘   └────────────┬─────────────┘
        └─────────────┬───────────┘
                       ▼
             ┌───────────────────┐
             │ generate_node       │  builds a labelled context block from
             │                     │  the retrieved chunks, picks one of two
             │                     │  system prompts (vector: "answer only
             │                     │  from these excerpts"; graph: "these ARE
             │                     │  the related articles, summarise them"),
             │                     │  and calls the LLM
             └──────────┬──────────┘
                        ▼
         {answer, route, chunks}  ──▶  returned to whichever
                                        interface called it
```

### Example 1 — a definitional question → vector route

> *"What are the lawful bases for processing personal data?"*

1. `classify_route` finds no article number + relational cue → **`vector`**.
2. `retrieve_node` embeds the question and runs
   `Chroma.similarity_search_with_score(query, k=4)`, returning the 4 nearest
   paragraph chunks: `Art. 6(1)` (×2), `Art. 5(1)`, `Art. 6(3)`.
3. `generate_node` uses the vector `SYSTEM_PROMPT` ("answer using ONLY the
   provided excerpts, cite as `[Art. N(p)]`"). The LLM produces:

   > *The lawful bases for processing personal data are as follows: 1. The
   > data subject has given consent... [Art. 6(1)] 2. Processing is necessary
   > for the performance of a contract... [Art. 6(1)] ...*

   — all six Art. 6(1)(a)–(f) bases, each correctly cited.

### Example 2 — a relationship question → graph route

> *"Which articles reference Article 6?"*

1. `classify_route` sees "Article 6" + "reference" → **`graph`**.
2. `graph_retrieve_node` runs the Cypher traversal in
   [graph_retriever.py](src/rag/graph_retriever.py): outgoing/incoming
   `REFERENCES` plus `IN_CHAPTER` siblings of Article 6, sorted by article
   number, capped at `k`. Returns 4 whole-article chunks: `Art. 5`, `Art. 7`,
   `Art. 8`, `Art. 9` (Chapter II siblings and cross-references — not just
   direct references, which is why the lowest-numbered neighbours win the cap
   rather than only articles that literally cite "Article 6").
3. `generate_node` switches to `GRAPH_SYSTEM_PROMPT` ("the context IS the set
   of related articles — summarise them"). The LLM produces:

   > *The articles that reference Article 6 are: 1. **[Art. 5]** Principles
   > relating to processing... 2. **[Art. 8]** Conditions applicable to
   > child's consent... 3. **[Art. 9]** Processing of special categories...*

   Note: in this real run the model covered 3 of the 4 retrieved sources
   (skipped `Art. 7`) — a known, documented limitation of the graph route's
   summarisation (see "Answer faithfulness" in
   [docs/evaluation-results.md](docs/evaluation-results.md)), not a retrieval
   error. The citations it does give are accurate; it just isn't exhaustive.

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
      "cwd": "<abs path to gdpr-smart-agent>",
      "env": { "PYTHONPATH": "<abs path to gdpr-smart-agent>" }
    }
  }
}
```
Four tools: `search_gdpr` (vector), `related_articles` (graph), `ask_gdpr`
(full routed agent, generated with **this server's** OpenAI/Anthropic key),
and `ask_gdpr_client` (same routing + retrieval, but generation is delegated
to the **calling client's own LLM** via MCP sampling — no server-side LLM
spend, and the answer comes back in the client's own model. Requires a client
with sampling support, e.g. Claude Desktop).

The explicit `PYTHONPATH` above isn't optional decoration — without it, some
Claude Desktop builds fail to import `src.mcp.server` (`MCP error -32000:
Connection closed`) because `-m src.mcp.server` needs the project root on
`sys.path`, and `cwd` alone isn't always enough to guarantee that.

<details>
<summary>Windows troubleshooting (Store/MSIX-packaged Claude Desktop)</summary>

If Claude Desktop was installed via the Microsoft Store rather than the
standalone installer, the setup differs from the official docs:

- **Config file isn't at `%APPDATA%\Claude\claude_desktop_config.json`.** It's
  under the package's virtualized folder instead:
  `%LOCALAPPDATA%\Packages\<Claude package name>\LocalCache\Roaming\Claude\claude_desktop_config.json`
  (find `<Claude package name>` with
  `dir "%LOCALAPPDATA%\Packages" | findstr Claude`). That file starts with no
  `mcpServers` key at all — add one.
- **Settings → Connectors is for remote (HTTPS URL) servers only.** Local
  stdio servers like this one go through **Settings → Developer → Edit
  Config** instead.
- **Single-instance lock**: closing the window doesn't fully quit the app, so
  relaunching after editing the config often just hands off to the still-running
  old process (silently keeping the stale config). If Developer settings still
  show "No servers added" after a restart, fully kill all `Claude.exe`
  processes first, then relaunch.
- **Cold start can exceed the app's ~10s first-connect timeout** — this
  server's heavy import chain (langchain/langgraph/chromadb/neo4j) can take
  10+ seconds on first launch. The app logs a timeout warning but usually
  recovers a few seconds later ("late stdio connect after announce"); check
  `%LOCALAPPDATA%\Packages\<Claude package name>\LocalCache\Roaming\Claude\logs\main.log`
  if it doesn't.

</details>

**Frontend** (Streamlit chat UI) — talks to the HTTP API, so start that first:
```bash
uvicorn src.api:app --reload            # terminal 1
streamlit run src/streamlit_app.py      # terminal 2, opens http://localhost:8501
```
Shows chat history, the retriever route (vector/graph) per answer, cited sources,
and a `k` slider in the sidebar. Point it at a different API host via the
sidebar's "API base URL" field (e.g. the Dockerized API on `:8000`).

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
