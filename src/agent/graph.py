"""Assemble and run the GDPR agent graph.

Flow:  START -> route -> (vector_retrieve | graph_retrieve) -> generate -> END

The router inspects the question and picks vector search (default) or graph
traversal (relationship questions naming specific articles). Both branches
produce the same ``chunks`` shape, so generate is source-agnostic.
"""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    generate_node,
    graph_retrieve_node,
    retrieve_node,
    route_node,
)
from src.agent.state import AgentState

load_dotenv()


def build_graph():
    """Build and compile the agent graph with vector/graph routing."""
    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("vector_retrieve", retrieve_node)
    builder.add_node("graph_retrieve", graph_retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"vector": "vector_retrieve", "graph": "graph_retrieve"},
    )
    builder.add_edge("vector_retrieve", "generate")
    builder.add_edge("graph_retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return a cached compiled graph (build once per process)."""
    return build_graph()


def answer_question(question: str, k: int | None = None) -> AgentState:
    """Run the full graph for ``question`` and return the final state."""
    inputs: AgentState = {"question": question}
    if k is not None:
        inputs["k"] = k
    return get_graph().invoke(inputs)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What are the lawful bases for processing personal data?"
    result = answer_question(q)

    print(f"Q: {q}")
    print(f"[route: {result.get('route')}]\n")
    print(result["answer"])
    print("\n--- Sources ---")
    for chunk in result.get("chunks", []):
        print(f"[{chunk.citation}] {chunk.title} - {chunk.url}")
