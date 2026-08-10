"""Tests for the agent nodes and compiled graph (LLM + retrieval mocked)."""

from neo4j.exceptions import ServiceUnavailable

from src.agent import nodes
from src.agent.graph import answer_question
from src.rag.vector_retriever import RetrievedChunk


def _chunk(article=6, para=1, text="Processing shall be lawful.") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        article_number=article,
        paragraph_number=para,
        title="Lawfulness of processing",
        chapter="Chapter II",
        url="https://gdpr-info.eu/art-6-gdpr/",
    )


class _FakeLLM:
    """Stand-in chat model that records the prompt and returns canned text."""

    def __init__(self, reply="Answer [Art. 6(1)]."):
        self.reply = reply
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return type("Resp", (), {"content": self.reply})()


# --- retrieve_node ---------------------------------------------------------

def test_retrieve_node_uses_default_k(monkeypatch):
    captured = {}
    monkeypatch.setattr(nodes, "retrieve", lambda q, k: captured.update(q=q, k=k) or [])
    nodes.retrieve_node({"question": "Q"})
    assert captured == {"q": "Q", "k": nodes.DEFAULT_TOP_K}


def test_retrieve_node_honours_explicit_k(monkeypatch):
    captured = {}
    monkeypatch.setattr(nodes, "retrieve", lambda q, k: captured.update(k=k) or [])
    nodes.retrieve_node({"question": "Q", "k": 9})
    assert captured["k"] == 9


# --- graph_retrieve_node ----------------------------------------------------

def test_graph_retrieve_node_falls_back_on_empty_result(monkeypatch):
    monkeypatch.setattr(nodes.graph_retriever, "retrieve", lambda q, limit: [])
    monkeypatch.setattr(nodes, "retrieve", lambda q, k: [_chunk()])

    out = nodes.graph_retrieve_node({"question": "Which articles reference Article 6?", "k": 4})

    assert [c.citation for c in out["chunks"]] == ["Art. 6(1)"]


def test_graph_retrieve_node_falls_back_when_neo4j_unreachable(monkeypatch):
    """A connectivity failure (Neo4j down) must degrade to vector search, not raise."""

    def _raise(question, limit):
        raise ServiceUnavailable("connection refused")

    monkeypatch.setattr(nodes.graph_retriever, "retrieve", _raise)
    monkeypatch.setattr(nodes, "retrieve", lambda q, k: [_chunk()])

    out = nodes.graph_retrieve_node({"question": "Which articles reference Article 6?", "k": 4})

    assert [c.citation for c in out["chunks"]] == ["Art. 6(1)"]


# --- generate_node ---------------------------------------------------------

def test_generate_node_without_chunks_returns_fallback():
    out = nodes.generate_node({"question": "Q", "chunks": []})
    assert "couldn't find" in out["answer"].lower()


def test_generate_node_builds_context_and_returns_answer(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(nodes, "_get_llm", lambda: fake)

    out = nodes.generate_node({"question": "Is it lawful?", "chunks": [_chunk()]})

    assert out["answer"] == "Answer [Art. 6(1)]."
    # The retrieved passage and its citation must reach the model.
    human = fake.last_messages[-1].content
    assert "[Art. 6(1)]" in human
    assert "Processing shall be lawful." in human
    assert "Is it lawful?" in human


# --- full graph ------------------------------------------------------------

def test_answer_question_end_to_end(monkeypatch):
    monkeypatch.setattr(nodes, "retrieve", lambda q, k: [_chunk(), _chunk(para=2)])
    monkeypatch.setattr(nodes, "_get_llm", lambda: _FakeLLM("Grounded answer [Art. 6(1)]."))

    result = answer_question("What is lawful processing?")

    assert result["answer"] == "Grounded answer [Art. 6(1)]."
    assert [c.citation for c in result["chunks"]] == ["Art. 6(1)", "Art. 6(2)"]
