# Evaluation Results

Reproduce with:

```bash
python -m scripts.evaluate
```

The harness ([scripts/evaluate.py](../scripts/evaluate.py)) scores a labelled
set ([data/eval/eval_questions.json](../data/eval/eval_questions.json)) on two
axes. Neither metric calls the answer-generation LLM, so runs are cheap and
stable; only the embedding model is used for retrieval.

## 1. Vector retrieval quality

For each definitional question with a known target article, we check whether
that article appears among the top-`k` retrieved chunks (`k = 4`, n = 15).

| Metric | Score |
|--------|-------|
| hit@1  | 0.87  |
| hit@3  | 0.93  |
| hit@4  | 0.93  |
| MRR    | 0.90  |

**Reading:** for 87% of questions the single most relevant chunk is already from
the correct article; 93% have it within the top 3.

### Misses

- **Art. 44** ("general principle for transfers") — retrieved Articles 45, 49,
  46 instead. These are the neighbouring transfer-mechanism articles in the same
  chapter; the general-principle article was edged out. A relationship question
  here would route to the graph and surface Art. 44's neighbours directly.

## 2. Router accuracy

Does `classify_route()` send each question to the intended retriever
(`vector` vs `graph`)? Measured over all 19 labelled questions — no API calls.

| Metric | Score |
|--------|-------|
| accuracy | 1.00 |

All definitional questions routed to vector search; all relationship questions
("which articles reference Article 6?", "what is related to Article 17?") routed
to the graph.

## Notes & next steps

- The set is small (19 questions) and hand-labelled; treat numbers as a smoke
  signal, not a benchmark. Expand `eval_questions.json` to harden it.
- Retrieval-only metrics don't measure answer faithfulness. A natural follow-up
  is an LLM-as-judge pass over generated answers (groundedness / citation
  correctness), kept separate so the cheap retrieval eval stays in CI.
