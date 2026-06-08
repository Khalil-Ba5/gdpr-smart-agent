"""Tests for the vector retriever (no network / API calls)."""

from langchain_core.documents import Document

from src.rag import vector_retriever
from src.rag.vector_retriever import RetrievedChunk, retrieve


def _doc(**meta) -> Document:
    base = {
        "article_number": 6,
        "paragraph_number": 1,
        "title": "Lawfulness of processing",
        "chapter": "Chapter II",
        "url": "https://gdpr-info.eu/art-6-gdpr/",
    }
    base.update(meta)
    return Document(page_content="Processing shall be lawful...", metadata=base)


def test_from_document_maps_metadata():
    chunk = RetrievedChunk.from_document(_doc(), score=0.12)
    assert chunk.article_number == 6
    assert chunk.paragraph_number == 1
    assert chunk.title == "Lawfulness of processing"
    assert chunk.url.endswith("art-6-gdpr/")
    assert chunk.score == 0.12


def test_citation_format():
    chunk = RetrievedChunk.from_document(_doc(article_number=17, paragraph_number=2))
    assert chunk.citation == "Art. 17(2)"


def test_from_document_tolerates_missing_metadata():
    chunk = RetrievedChunk.from_document(Document(page_content="x", metadata={}))
    assert chunk.article_number == "?"
    assert chunk.paragraph_number == "?"
    assert chunk.citation == "Art. ?(?)"


def test_retrieve_passes_k_and_wraps_results(monkeypatch):
    """retrieve() should forward k to Chroma and wrap docs as RetrievedChunks."""
    captured = {}

    class FakeStore:
        def similarity_search_with_score(self, query, k):
            captured["query"] = query
            captured["k"] = k
            return [(_doc(), 0.1), (_doc(paragraph_number=2), 0.2)]

    monkeypatch.setattr(vector_retriever, "get_vectorstore", lambda: FakeStore())

    chunks = retrieve("what is lawful processing?", k=5)

    assert captured == {"query": "what is lawful processing?", "k": 5}
    assert [c.citation for c in chunks] == ["Art. 6(1)", "Art. 6(2)"]
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
