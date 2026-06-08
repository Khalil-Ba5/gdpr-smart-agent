"""Build both indexes in one command: Chroma vector store + Neo4j graph.

Convenience wrapper so a fresh checkout only needs:

    docker compose up -d neo4j
    python -m scripts.seed_data

Requires OPENAI_API_KEY (embeddings) and a reachable Neo4j (NEO4J_* in .env).
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def seed_vectors() -> None:
    print("→ Building Chroma vector index...")
    from src.rag.ingest import build_chroma_index

    build_chroma_index()
    print("  ✓ vector index ready (chroma_db/)")


def seed_graph() -> None:
    print("→ Loading Neo4j knowledge graph...")
    from src.graph.load_graph import main as load_graph

    load_graph()
    print("  ✓ graph loaded")


def main() -> None:
    seed_vectors()
    seed_graph()
    print("\nDone. Both indexes are ready — start the API with:")
    print("    uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
