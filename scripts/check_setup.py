"""Preflight connectivity check for the GDPR Smart Agent.

Verifies the external services the app depends on are reachable before you run
ingest / load / the API. Unlike the unit tests (which mock everything), this
makes real connections, so run it after editing .env or (re)starting Neo4j.

    python -m scripts.check_setup
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OK = "[ ok ]"
FAIL = "[fail]"


def check_llm() -> bool:
    try:
        from langchain_openai import ChatOpenAI

        reply = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o")).invoke("Say hello")
        print(f"{OK} LLM (OpenAI): {reply.content[:30]!r}")
        return True
    except Exception as exc:
        print(f"{FAIL} LLM: {type(exc).__name__}: {exc}")
        return False


def check_vector_store() -> bool:
    try:
        from src.rag.vector_retriever import get_vectorstore

        count = get_vectorstore()._collection.count()
        print(f"{OK} Chroma: {count} embeddings in collection")
        return True
    except Exception as exc:
        print(f"{FAIL} Chroma: {type(exc).__name__}: {exc}  (run: python -m src.rag.ingest)")
        return False


def check_neo4j() -> bool:
    try:
        from src.graph.schema import get_driver

        driver = get_driver()
        driver.verify_connectivity()
        with driver.session() as session:
            n = session.run("MATCH (a:Article) RETURN count(a) AS c").single()["c"]
        driver.close()
        print(f"{OK} Neo4j: connected, {n} Article nodes")
        return True
    except Exception as exc:
        print(f"{FAIL} Neo4j: {type(exc).__name__}: {exc}  (run: docker compose up -d neo4j)")
        return False


def main() -> None:
    print("Checking external services...\n")
    results = [check_llm(), check_vector_store(), check_neo4j()]
    print()
    if all(results):
        print("All systems go.")
    else:
        print("Some checks failed — see messages above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
