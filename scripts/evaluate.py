"""Evaluate retrieval quality and router accuracy against a labelled set.

Two metrics, both cheap (no answer-generation LLM calls):

1. Vector retrieval — for definitional questions with a known target article,
   does that article appear in the top-k retrieved chunks? Reports hit@1/3/k
   and mean reciprocal rank (MRR).
2. Router accuracy — does classify_route() match the labelled route?

Usage:
    python -m scripts.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from src.agent.nodes import classify_route
from src.rag.vector_retriever import retrieve

load_dotenv()

EVAL_PATH = "data/eval/eval_questions.json"
K = 4


def _first_rank(expected: list[int], retrieved_articles: list[int]) -> int | None:
    """1-based rank of the first retrieved article that is in `expected`."""
    for rank, art in enumerate(retrieved_articles, start=1):
        if art in expected:
            return rank
    return None


def evaluate(eval_path: str = EVAL_PATH, k: int = K) -> dict:
    cases = json.loads(Path(eval_path).read_text(encoding="utf-8"))

    vector_cases = [c for c in cases if c["route"] == "vector" and c["expected_articles"]]
    hits = {1: 0, 3: 0, k: 0}
    reciprocal_ranks = []
    rows = []

    for case in vector_cases:
        chunks = retrieve(case["question"], k=k)
        retrieved = [c.article_number for c in chunks]
        rank = _first_rank(case["expected_articles"], retrieved)
        if rank:
            reciprocal_ranks.append(1 / rank)
            for threshold in hits:
                if rank <= threshold:
                    hits[threshold] += 1
        else:
            reciprocal_ranks.append(0.0)
        rows.append((case["expected_articles"], rank, retrieved))

    n = len(vector_cases)
    retrieval = {
        "n": n,
        "hit@1": hits[1] / n,
        "hit@3": hits[3] / n,
        f"hit@{k}": hits[k] / n,
        "mrr": sum(reciprocal_ranks) / n,
    }

    # Router accuracy (no API calls).
    correct = sum(1 for c in cases if classify_route(c["question"]) == c["route"])
    routing = {"n": len(cases), "accuracy": correct / len(cases)}

    return {"retrieval": retrieval, "routing": routing, "rows": rows, "cases": cases}


def main() -> None:
    res = evaluate()
    r = res["retrieval"]
    print(f"Vector retrieval (k={K}, n={r['n']})")
    print(f"  hit@1 = {r['hit@1']:.2f}   hit@3 = {r['hit@3']:.2f}   "
          f"hit@{K} = {r[f'hit@{K}']:.2f}   MRR = {r['mrr']:.3f}")
    print("\n  expected -> rank | retrieved articles")
    for expected, rank, retrieved in res["rows"]:
        print(f"  {str(expected):>8} -> {str(rank):>4} | {retrieved}")

    rt = res["routing"]
    print(f"\nRouter accuracy (n={rt['n']}) = {rt['accuracy']:.2f}")


if __name__ == "__main__":
    main()
