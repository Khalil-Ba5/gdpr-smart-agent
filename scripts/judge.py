"""LLM-as-judge evaluation of answer faithfulness.

For each question, runs the full agent and asks a judge model to score the
generated answer against the retrieved source passages on two axes:

- groundedness    : is every claim supported by the provided sources? (1-5)
- citation_quality: are claims cited to articles that appear in the sources? (1-5)

This is the expensive eval (it calls the answer LLM and the judge), so it is a
separate script from scripts/evaluate.py and runs on a small sample by default.

    python -m scripts.judge            # default sample
    python -m scripts.judge --limit 8  # smaller/cheaper
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.agent.graph import answer_question
from src.agent.nodes import _format_context, _get_llm

load_dotenv()

EVAL_PATH = "data/eval/eval_questions.json"
DEFAULT_SAMPLE = 12

JUDGE_SYSTEM = (
    "You are a strict evaluator of a GDPR assistant. You are given a question, "
    "the GDPR source passages that were retrieved, and the assistant's answer. "
    "Score the answer ONLY against the provided sources.\n"
    "Return a JSON object with exactly these keys:\n"
    '  "groundedness": integer 1-5 (5 = every claim is supported by the sources, '
    "1 = mostly unsupported/hallucinated),\n"
    '  "citation_quality": integer 1-5 (5 = claims are cited with [Art. N] labels '
    "that match the sources, 1 = missing or wrong citations),\n"
    '  "reason": one short sentence.\n'
    "Respond with JSON only, no prose."
)


def _judge(question: str, answer: str, context: str, judge_llm) -> dict:
    prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved sources:\n{context}\n\n"
        f"Assistant answer:\n{answer}\n\n"
        "Score it as instructed."
    )
    raw = judge_llm.invoke(
        [{"role": "system", "content": JUDGE_SYSTEM},
         {"role": "user", "content": prompt}]
    ).content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"groundedness": None, "citation_quality": None, "reason": "unparseable"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"groundedness": None, "citation_quality": None, "reason": "bad json"}
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    all_cases = json.loads(Path(EVAL_PATH).read_text(encoding="utf-8"))
    # Sample evenly across the file so both vector and graph questions appear.
    if args.limit >= len(all_cases):
        cases = all_cases
    else:
        idx = sorted({round(i * (len(all_cases) - 1) / (args.limit - 1)) for i in range(args.limit)})
        cases = [all_cases[i] for i in idx]
    judge_llm = _get_llm()

    ground_scores, cite_scores = [], []
    print(f"Judging {len(cases)} answers...\n")
    for case in cases:
        q = case["question"]
        result = answer_question(q)
        context = _format_context(result.get("chunks", []))
        scores = _judge(q, result["answer"], context, judge_llm)

        g, c = scores.get("groundedness"), scores.get("citation_quality")
        if isinstance(g, int):
            ground_scores.append(g)
        if isinstance(c, int):
            cite_scores.append(c)
        print(f"[{result.get('route'):>6}] g={g} c={c}  {q[:60]}")

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n=== LLM-as-judge summary ===")
    print(f"n scored          : {len(ground_scores)}")
    print(f"groundedness avg  : {avg(ground_scores):.2f} / 5")
    print(f"citation_quality  : {avg(cite_scores):.2f} / 5")


if __name__ == "__main__":
    main()
