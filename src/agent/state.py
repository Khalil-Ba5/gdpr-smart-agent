"""Shared state for the GDPR agent's LangGraph."""

from __future__ import annotations

from typing import TypedDict

from src.rag.vector_retriever import RetrievedChunk


class AgentState(TypedDict, total=False):
    """State threaded through the agent graph.

    ``question`` is the only required input. ``k`` optionally overrides how
    many chunks to retrieve. The retrieve node fills ``chunks``; the generate
    node fills ``answer``.
    """

    question: str
    k: int
    route: str  # "vector" | "graph" — chosen by the router node
    chunks: list[RetrievedChunk]
    answer: str
