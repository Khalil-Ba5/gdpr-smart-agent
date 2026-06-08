"""MCP server exposing the GDPR retrievers and agent as tools.

Runs over stdio so it can be wired into MCP clients (e.g. Claude Desktop):

    {
      "mcpServers": {
        "gdpr": {
          "command": "C:/.../gdpr-smart-agent/venv/Scripts/python.exe",
          "args": ["-m", "src.mcp.server"],
          "cwd": "C:/.../gdpr-smart-agent"
        }
      }
    }

Tools
-----
- search_gdpr(query, k)        : semantic search over GDPR paragraphs (vector)
- related_articles(query, limit): articles linked to those named in the query (graph)
- ask_gdpr(question)           : full routed agent answer with citations
"""

from __future__ import annotations

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.agent.graph import answer_question
from src.rag import graph_retriever
from src.rag.vector_retriever import RetrievedChunk, retrieve

load_dotenv()

mcp = FastMCP("gdpr-smart-agent")


def _serialize(chunk: RetrievedChunk) -> dict:
    return {
        "citation": chunk.citation,
        "article_number": chunk.article_number,
        "paragraph_number": chunk.paragraph_number,
        "title": chunk.title,
        "url": chunk.url,
        "text": chunk.text,
    }


@mcp.tool()
def search_gdpr(query: str, k: int = 4) -> list[dict]:
    """Semantic search over GDPR article paragraphs.

    Use for definitional or substantive questions ("what is personal data?").
    Returns the k most relevant passages with article/paragraph citations.
    """
    return [_serialize(c) for c in retrieve(query, k=k)]


@mcp.tool()
def related_articles(query: str, limit: int = 6) -> list[dict]:
    """Find GDPR articles related to those named in the query.

    Use for relationship questions ("what articles reference Article 6?").
    Traverses cross-references and chapter siblings in the knowledge graph.
    Requires the Neo4j graph to be running and loaded.
    """
    return [_serialize(c) for c in graph_retriever.retrieve(query, limit=limit)]


@mcp.tool()
def ask_gdpr(question: str) -> dict:
    """Answer a GDPR question with citations using the full routed agent.

    The agent automatically picks vector or graph retrieval based on the
    question, then generates a grounded, cited answer.
    """
    result = answer_question(question)
    return {
        "answer": result["answer"],
        "route": result.get("route"),
        "sources": [_serialize(c) for c in result.get("chunks", [])],
    }


def main() -> None:
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
