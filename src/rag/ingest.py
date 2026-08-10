import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loads OPENAI_API_KEY (and friends) from .env — every other entry point does
# this; running `python -m src.rag.ingest` directly (as the README documents)
# previously depended on OPENAI_API_KEY already being exported in the shell.
load_dotenv()

JSON_PATH = "data/docs/gdpr_articles.json"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "gdpr_articles"


def load_gdpr_documents(json_path: str) -> list[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []

    # Better for legal RAG: one document per paragraph, with rich metadata
    for article in data:
        article_number = article["article_number"]
        title = article.get("title", "")
        chapter = article.get("chapter", "")
        recitals = article.get("recitals", [])
        url = article.get("url", "")

        for paragraph in article.get("paragraphs", []):
            docs.append(
                Document(
                    page_content=paragraph["text"],
                    metadata={
                        "article_number": article_number,
                        "paragraph_number": paragraph["number"],
                        "title": title,
                        "chapter": chapter,
                        "recitals": ",".join(str(r) for r in recitals),
                        "url": url,
                    },
                )
            )

    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )
    return splitter.split_documents(documents)


def build_chroma_index() -> Chroma:
    raw_docs = load_gdpr_documents(JSON_PATH)
    chunked_docs = split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    # Creates or reuses a persistent Chroma collection on disk
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # Add stable IDs so repeated runs are easier to manage
    ids = []
    for i, doc in enumerate(chunked_docs):
        article_no = doc.metadata.get("article_number", "unknown")
        para_no = doc.metadata.get("paragraph_number", "unknown")
        ids.append(f"article-{article_no}-para-{para_no}-chunk-{i}")

    vectorstore.add_documents(documents=chunked_docs, ids=ids)

    return vectorstore


if __name__ == "__main__":
    db = build_chroma_index()
    print("Done.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Persisted at: {PERSIST_DIR}")

    # Quick test query
    results = db.similarity_search("What is personal data?", k=3)
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(doc.metadata)
        print(doc.page_content[:300])