"""Nodes for the GDPR agent graph: route -> retrieve (vector|graph) -> generate."""

from __future__ import annotations

import logging
import os
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from neo4j.exceptions import GqlError  # common base for Neo4jError (server) + DriverError (connectivity)

from src.agent.state import AgentState
from src.rag import graph_retriever
from src.rag.vector_retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

# Default to OpenAI to match the existing .env / ingest setup. Swap to Claude by
# setting LLM_PROVIDER=anthropic (and ANTHROPIC_API_KEY) — see _get_llm below.
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_TOP_K = 4

# Words that signal a *relationship* question better served by the graph.
_RELATIONAL_HINT = re.compile(
    r"\b(relate[ds]?|relation|referenc\w*|cross-?referenc\w*|connect\w*|"
    r"link\w*|associat\w*|depend\w*|interact\w*|related to|same chapter|"
    r"what else|which other)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a GDPR legal assistant. Answer the user's question using ONLY the "
    "GDPR excerpts provided in the context. Cite the specific article and "
    "paragraph for every claim, using the bracketed citation labels exactly as "
    "they appear in the context (e.g. [Art. 6(1)]). If the context does not "
    "contain the answer, say so plainly rather than guessing. Be concise and "
    "precise; do not invent article numbers."
)

# Used on the graph route: the context IS the set of related articles, so the
# model should describe those links rather than refuse for lack of the anchor.
GRAPH_SYSTEM_PROMPT = (
    "You are a GDPR legal assistant. The user is asking which GDPR provisions "
    "relate to a specific article. The context lists exactly those related "
    "articles (connected by cross-reference or shared chapter). Summarise them "
    "and briefly explain how each connects to the article in question, citing "
    "each as it is labelled (e.g. [Art. 7]). The related articles ARE the ones "
    "provided — do not say you lack the information. Be concise."
)


def _get_llm() -> BaseChatModel:
    """Build the chat model, honoring LLM_PROVIDER / *_MODEL env overrides."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=0,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    blocks = []
    for chunk in chunks:
        header = f"[{chunk.citation}] {chunk.title}".strip()
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def classify_route(question: str) -> str:
    """Decide which retriever fits the question: ``"graph"`` or ``"vector"``.

    A question routes to the graph only when it both names specific article(s)
    *and* asks about relationships between provisions (e.g. "which articles
    reference Article 6?"). Everything else uses vector search.
    """
    has_article = bool(graph_retriever.parse_article_numbers(question))
    is_relational = bool(_RELATIONAL_HINT.search(question or ""))
    return "graph" if (has_article and is_relational) else "vector"


def route_node(state: AgentState) -> AgentState:
    """Record the routing decision so the graph can branch on it."""
    return {"route": classify_route(state["question"])}


def retrieve_node(state: AgentState) -> AgentState:
    """Vector retrieval: fetch the most relevant GDPR passages for the question."""
    k = state.get("k") or DEFAULT_TOP_K
    chunks = retrieve(state["question"], k=k)
    return {"chunks": chunks}


def graph_retrieve_node(state: AgentState) -> AgentState:
    """Graph retrieval: expand to articles related to those named in the question.

    Falls back to vector retrieval if the graph yields nothing (e.g. the
    referenced article has no recorded neighbours) *or* if the graph itself is
    unreachable (Neo4j down/misconfigured) — a connectivity failure should
    degrade the answer, not crash the request.
    """
    k = state.get("k") or DEFAULT_TOP_K
    try:
        chunks = graph_retriever.retrieve(state["question"], limit=k)
    except GqlError:
        logger.warning("Graph retrieval failed; falling back to vector search.", exc_info=True)
        chunks = []
    if not chunks:
        chunks = retrieve(state["question"], k=k)
    return {"chunks": chunks}


def generate_node(state: AgentState) -> AgentState:
    """Generate a cited answer grounded in the retrieved passages."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"answer": "I couldn't find any relevant GDPR provisions for that question."}

    # On the graph route the context *is* the set of related articles, so use a
    # prompt that frames it that way — otherwise the model says it "can't find"
    # the link to an anchor article whose text wasn't retrieved.
    is_graph = state.get("route") == "graph"
    system_prompt = GRAPH_SYSTEM_PROMPT if is_graph else SYSTEM_PROMPT
    intro = "Related articles:" if is_graph else "Context:"

    context = _format_context(chunks)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{intro}\n{context}\n\nQuestion: {state['question']}"),
    ]
    response = _get_llm().invoke(messages)
    return {"answer": response.content}
