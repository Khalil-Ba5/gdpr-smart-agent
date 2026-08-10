"""Vector retrieval over the persisted Chroma index.

The index is built by ``src/rag/ingest.py`` (one chunk per GDPR paragraph,
embedded with OpenAI ``text-embedding-3-small``). This module loads that same
collection read-only and exposes a small ``retrieve`` helper that returns the
most relevant chunks together with their citation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Must match the values used in src/rag/ingest.py so we open the same index.
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "gdpr_articles"
EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class RetrievedChunk:
    """A single retrieved passage plus the metadata needed to cite it."""

    text: str
    article_number: int
    paragraph_number: int | str
    title: str
    chapter: str
    url: str
    score: float | None = None

    @property
    def citation(self) -> str:
        """Short human-readable citation, e.g. ``Art. 6(1)`` or ``Art. 6``.

        Whole-article results (e.g. from the graph retriever) have no
        paragraph number and render as ``Art. N``.
        """
        if self.paragraph_number == "" or self.paragraph_number is None:
            return f"Art. {self.article_number}"
        return f"Art. {self.article_number}({self.paragraph_number})"

    @classmethod
    def from_document(cls, doc: Document, score: float | None = None) -> "RetrievedChunk":
        meta = doc.metadata or {}
        return cls(
            text=doc.page_content,
            article_number=meta.get("article_number", "?"),
            paragraph_number=meta.get("paragraph_number", "?"),
            title=meta.get("title", ""),
            chapter=meta.get("chapter", ""),
            url=meta.get("url", ""),
            score=score,
        )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Open the persisted Chroma collection (read-only use).

    Cached per process — reopening the embeddings client + SQLite-backed store
    on every call is wasted I/O once this is called per-request (API/MCP).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )


def retrieve(query: str, k: int = 4) -> list[RetrievedChunk]:
    """Return the ``k`` most relevant GDPR chunks for ``query``.

    Scores are Chroma distances (lower = closer); they are attached for
    debugging/ranking but are not required for answer generation.
    """
    store = get_vectorstore()
    results = store.similarity_search_with_score(query, k=k)
    return [RetrievedChunk.from_document(doc, score) for doc, score in results]


if __name__ == "__main__":
    for chunk in retrieve("What is personal data?", k=3):
        print(f"\n[{chunk.citation}] {chunk.title}  (score={chunk.score:.4f})")
        print(chunk.text[:200])
