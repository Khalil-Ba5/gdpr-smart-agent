"""Tests for the MCP server's sampling-based tool (LLM + retrieval mocked).

ask_gdpr/search_gdpr/related_articles are thin passthroughs already covered
indirectly via src.rag / src.agent tests. ask_gdpr_client has its own logic
(retrieval + prompt selection + delegating generation via MCP sampling) worth
testing directly. This was also verified manually against the real MCP
protocol with a fake sampling client (see project chat history) — this test
locks that behaviour in without needing a live subprocess.
"""

import pytest

from src.mcp import server
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


class _FakeMessageResult:
    def __init__(self, text="Client-generated answer [Art. 6(1)].", model="fake-client-llm"):
        self.content = server.TextContent(type="text", text=text)
        self.model = model


class _FakeSession:
    """Records the sampling request and returns a canned completion."""

    def __init__(self, reply=None):
        self.reply = reply or _FakeMessageResult()
        self.last_call = None

    async def create_message(self, **kwargs):
        self.last_call = kwargs
        return self.reply


class _FakeContext:
    def __init__(self, session):
        self.session = session


@pytest.mark.anyio
async def test_ask_gdpr_client_delegates_generation_via_sampling(monkeypatch):
    monkeypatch.setattr(server, "retrieve_node", lambda state: {"chunks": [_chunk()]})
    session = _FakeSession()
    ctx = _FakeContext(session)

    result = await server.ask_gdpr_client("What is lawful processing?", ctx)

    # Generation came back from the (fake) client, not a server-side LLM call.
    assert result["answer"] == "Client-generated answer [Art. 6(1)]."
    assert result["route"] == "vector"
    assert result["generated_by"] == "fake-client-llm"
    assert [s["citation"] for s in result["sources"]] == ["Art. 6(1)"]

    # The sampling request must carry the retrieved context + question.
    assert session.last_call["system_prompt"] == server.SYSTEM_PROMPT
    sent_text = session.last_call["messages"][0].content.text
    assert "Processing shall be lawful." in sent_text
    assert "What is lawful processing?" in sent_text


@pytest.mark.anyio
async def test_ask_gdpr_client_uses_graph_prompt_on_graph_route(monkeypatch):
    monkeypatch.setattr(server, "graph_retrieve_node", lambda state: {"chunks": [_chunk()]})
    session = _FakeSession()
    ctx = _FakeContext(session)

    result = await server.ask_gdpr_client("Which articles reference Article 6?", ctx)

    assert result["route"] == "graph"
    assert session.last_call["system_prompt"] == server.GRAPH_SYSTEM_PROMPT
    assert session.last_call["messages"][0].content.text.startswith("Related articles:")


@pytest.mark.anyio
async def test_ask_gdpr_client_skips_sampling_when_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(server, "retrieve_node", lambda state: {"chunks": []})
    session = _FakeSession()
    ctx = _FakeContext(session)

    result = await server.ask_gdpr_client("What is lawful processing?", ctx)

    assert "couldn't find" in result["answer"].lower()
    assert result["sources"] == []
    assert session.last_call is None  # never asked the client to sample


@pytest.fixture
def anyio_backend():
    return "asyncio"
