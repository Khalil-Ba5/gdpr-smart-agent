"""Tests for the Neo4j graph-RAG layer (cross-ref extraction, retriever, router).

All network-free: the Neo4j driver is faked, so these run without a live DB.
"""

from contextlib import contextmanager

from src.agent.nodes import classify_route
from src.graph.extract_entities import extract_article_references
from src.rag import graph_retriever


# --- cross-reference extraction -------------------------------------------

def test_extract_single_and_list_references():
    text = "In accordance with Article 6 and Articles 13, 14 and 15, the controller..."
    assert extract_article_references(text) == [6, 13, 14, 15]


def test_extract_expands_ranges_and_excludes_self():
    text = "Transfers under Articles 44 to 47 are subject to Article 5."
    assert extract_article_references(text, source_article=5) == [44, 45, 46, 47]


def test_extract_ignores_paragraph_numbers():
    # "Article 6(1)" must yield article 6 only, never article 1.
    assert extract_article_references("point (a) of Article 6(1)") == [6]


# --- question parsing ------------------------------------------------------

def test_parse_article_numbers_from_question():
    assert graph_retriever.parse_article_numbers(
        "How do Article 6 and Art. 17 interact?"
    ) == [6, 17]


# --- router ----------------------------------------------------------------

def test_router_picks_graph_for_relationship_questions():
    assert classify_route("Which articles reference Article 6?") == "graph"
    assert classify_route("What is related to Article 17?") == "graph"


def test_router_picks_vector_without_article_or_relation():
    # Relationship words but no article number -> vector.
    assert classify_route("What articles are related to consent?") == "vector"
    # Article number but a plain definitional question -> vector.
    assert classify_route("What does Article 4 say about personal data?") == "vector"


# --- graph retriever (mocked driver) --------------------------------------

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows, calls):
        self._rows = rows
        self._calls = calls

    def run(self, query, **params):
        self._calls.append(params)
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    @contextmanager
    def session(self):
        yield _FakeSession(self._rows, self.calls)


def test_graph_retriever_returns_related_articles():
    rows = [
        {"number": 7, "title": "Conditions for consent", "full_text": "...", "url": "u7"},
        {"number": 8, "title": "Child's consent", "full_text": "...", "url": "u8"},
    ]
    driver = _FakeDriver(rows)

    chunks = graph_retriever.retrieve_related([6], limit=5, driver=driver)

    assert [c.citation for c in chunks] == ["Art. 7", "Art. 8"]
    assert driver.calls[0] == {"numbers": [6], "limit": 5}


def test_graph_retriever_empty_for_no_anchors():
    driver = _FakeDriver([])
    assert graph_retriever.retrieve_related([], driver=driver) == []
    assert driver.calls == []  # short-circuits before querying
