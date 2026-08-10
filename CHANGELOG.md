# Changelog

## 2026-08-10 — Verified live in Claude Desktop; documented Windows setup gotchas

Connected the MCP server to a real Claude Desktop instance end to end (not
just the test harness from the previous entry) and fixed what broke along the
way:

- Added `PYTHONPATH` to the example MCP client config
  ([README.md](README.md)) — without it, `-m src.mcp.server` failed instantly
  on launch (`MCP error -32000: Connection closed`) because `cwd` alone wasn't
  reliably enough for `sys.path` resolution in this environment.
- Documented Windows Store/MSIX-packaged Claude Desktop quirks in a
  collapsible README troubleshooting section: the real (non-standard) config
  file path, the fact that Settings → Connectors is remote-only (local stdio
  servers need Settings → Developer → Edit Config instead), the single-instance
  relaunch trap, and the ~10s cold-start timeout this server's import chain
  can exceed on first launch.
- End state, confirmed from the app's own logs: `Connected to gdpr (4 tools)`.

## 2026-08-09 — MCP sampling: delegate generation to the client's own LLM

Added a fourth MCP tool, `ask_gdpr_client` ([src/mcp/server.py](src/mcp/server.py)),
alongside the existing `search_gdpr`/`related_articles`/`ask_gdpr`. Routing and
retrieval are identical to `ask_gdpr` (reuses `classify_route`, `retrieve_node`,
`graph_retrieve_node`, and the same `SYSTEM_PROMPT`/`GRAPH_SYSTEM_PROMPT` from
`src/agent/nodes.py`) — the difference is generation: instead of calling this
server's own LLM (`_get_llm()`), it sends the retrieved context to the
**calling client** via MCP sampling (`ctx.session.create_message`) and lets
the client's own model write the answer. No duplicate LLM spend, and the
answer comes back in the connected assistant's own model/voice.

Verified two ways:
- **Real protocol, fake client LLM**: spawned the actual server over stdio,
  supplied a `sampling_callback` standing in for Claude Desktop's model, and
  called `ask_gdpr_client` end to end. Confirmed via server logs that
  retrieval made one real OpenAI *embeddings* call but **zero completions
  calls** — generation genuinely never touched the server's LLM key.
- **Unit tests** ([tests/test_mcp_server.py](tests/test_mcp_server.py)) —
  mocks `ctx.session.create_message` directly; covers the vector route, the
  graph route (correct prompt selection), and the empty-retrieval short
  circuit (never calls sampling if there's nothing to answer from). 22/22
  tests passing project-wide (19 prior + 3 new).

## 2026-08-09 — Reliability, perf, and hygiene fixes

Full review of every component (scraper → ingest → vector/graph retrieval →
agent → API/MCP/frontend → scripts → tests → config) turned up one real bug
and a set of smaller correctness, performance, and hygiene issues. All are
fixed below and verified: `python -m pytest` — 19/19 passing (17 pre-existing
+ 2 new regression tests).

### Fixed

- **Graph route crashed instead of falling back when Neo4j was unreachable**
  ([src/agent/nodes.py](src/agent/nodes.py)) — `graph_retrieve_node` only
  handled the *empty-result* case (graph up, but no neighbours found); a
  connectivity failure (`neo4j.exceptions.ServiceUnavailable`, e.g. Neo4j not
  running) propagated uncaught, 500-ing the API and crashing MCP calls,
  contradicting the documented fallback behaviour. Now catches
  `neo4j.exceptions.GqlError` (the common base for both server-side and
  connectivity errors), logs a warning, and falls through to vector search —
  reproduced and verified fixed against a live unreachable Neo4j instance.
  Regression tests added in [tests/test_agent.py](tests/test_agent.py).

- **`src/rag/ingest.py` never loaded `.env`** — every other entry point calls
  `load_dotenv()`; this one didn't, so running `python -m src.rag.ingest`
  directly (as the README documents) silently used no `OPENAI_API_KEY` unless
  it was already exported in the shell. Fixed.

- **Latent `KeyError` in `ingest.py`** — `load_gdpr_documents` computed
  `recitals = article.get("recitals", [])` (safe default) but then used
  `article["recitals"]` (direct index) when building chunk metadata, so an
  article missing the `recitals` key would crash ingestion instead of
  defaulting to `[]`. Now uses the already-computed, safe `recitals` variable.

### Performance

- **Chroma vectorstore reconnected on every retrieval call**
  ([src/rag/vector_retriever.py](src/rag/vector_retriever.py)) —
  `get_vectorstore()` re-instantiated `OpenAIEmbeddings` and reopened the
  SQLite-backed Chroma store per call. Now cached per-process with
  `@lru_cache`, matching the caching pattern already used for the compiled
  agent graph.

- **Neo4j driver opened and closed on every graph query**
  ([src/graph/schema.py](src/graph/schema.py),
  [src/rag/graph_retriever.py](src/rag/graph_retriever.py)) — added
  `get_cached_driver()`, a process-wide singleton driver for the retrieval hot
  path (Neo4j drivers are meant to be long-lived and pool connections
  internally). `get_driver()` is unchanged and still used by the one-off
  scripts (`load_graph.py`, `check_setup.py`) that own and close their own
  driver.

### Hardening

- **FastAPI now has a global exception handler**
  ([src/api.py](src/api.py)) — unhandled errors return a clean `500 {"detail":
  "Internal server error."}` instead of leaking a stack trace to the client;
  the full exception is still logged server-side.

- **MCP `related_articles` tool** ([src/mcp/server.py](src/mcp/server.py)) —
  this tool is graph-only by design (unlike `ask_gdpr`, which already
  degrades gracefully via the agent's fallback), so a Neo4j connectivity
  failure now raises a clear `RuntimeError` telling the caller how to fix it,
  instead of a raw low-level driver traceback.

- **Scraper** ([scripts/scrape_gdpr.py](scripts/scrape_gdpr.py)) — added a
  0.5s delay between the 99 sequential requests to gdpr-info.eu (previously
  none — risked rate-limiting/blocks on re-scrape), and a warning when a
  scraped article comes back with no title/paragraphs (previously silent —
  a site markup change would produce empty articles with no signal).

### Cleanup

- Removed `pydantic-settings` from `requirements.txt` — listed as a
  dependency but never imported anywhere; config goes through `os.getenv()` +
  `python-dotenv` throughout.
- Removed the empty, git-untracked, unreferenced `data/graph/` directory.
- Updated `.env.example`'s Neo4j comment — it said "not yet wired up," which
  was stale; the graph-RAG layer has been fully implemented and routed to
  since the second commit.
- Updated the default Anthropic model id (`DEFAULT_ANTHROPIC_MODEL` in
  `src/agent/nodes.py`, and `.env.example`) from `claude-sonnet-4-6` to
  `claude-sonnet-5`.
- Added `.github/workflows/ci.yml` — runs the fully network-free `pytest`
  suite (no API keys/services needed) on every push/PR. Deliberately does
  **not** run `scripts/evaluate.py` in CI: it needs a prebuilt `chroma_db/`
  index (gitignored, not reproducible from a clean checkout) and a real
  `OPENAI_API_KEY`, i.e. it costs money and needs a secret — that's a decision
  left to you rather than made silently. Run it locally; see
  [docs/evaluation-results.md](docs/evaluation-results.md).

### Not fixed (deliberately out of scope)

- No auth/rate-limiting was added to the `/ask` endpoint — this needs a design
  decision (API key? OAuth? just a reverse-proxy?) that's yours to make, not
  something to bolt on silently.
