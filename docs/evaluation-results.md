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
that article appears among the top-`k` retrieved chunks (`k = 4`, n = 45,
spanning 45 distinct articles across all GDPR chapters).

| Metric | Score |
|--------|-------|
| hit@1  | 0.62  |
| hit@3  | 0.84  |
| hit@4  | 0.91  |
| MRR    | 0.735 |

**Reading:** the correct article is the single best match 62% of the time and
within the top 4 for 91% of questions. (An earlier 15-question set scored
hit@1 0.87 — the broader, harder set here is a more honest signal.)

### Misses (article not in top 4)

- **Art. 2** ("material scope") — generic phrasing ("scope of the GDPR")
  retrieves scope-adjacent articles (23, 24) instead.
- **Art. 10** ("criminal convictions data") — edged out by the related special
  -category / safeguards articles (37, 35, 6, 27).
- **Art. 16** ("right to rectification") — confused with restriction (18) and
  remedies (79, 80) articles.
- **Art. 44** ("general principle for transfers") — retrieved the neighbouring
  transfer-mechanism articles (45, 46, 49) in the same chapter.

These are mostly cases where several articles in one chapter are semantically
close. A relationship question about any of them routes to the graph and surfaces
the neighbours directly; query expansion or reranking would help the rest.

## 2. Router accuracy

Does `classify_route()` send each question to the intended retriever
(`vector` vs `graph`)? Measured over all 53 labelled questions — no API calls.

| Metric | Score |
|--------|-------|
| accuracy | 1.00 |

All definitional questions routed to vector search; all relationship questions
("which articles reference Article 6?", "what is related to Article 17?") routed
to the graph.

## 3. Answer faithfulness (LLM-as-judge)

Reproduce with:

```bash
python -m scripts.judge --limit 12
```

[scripts/judge.py](../scripts/judge.py) runs the full agent on a sample of
questions (spread across both routes), then asks a judge model to score each
answer **only against its retrieved sources** on two 1-5 axes. This is the
expensive eval — it calls both the answer LLM and the judge — so it is kept
separate from the retrieval eval and runs on a sample.

Sample of 12 (10 vector + 2 graph):

| Metric | Score (avg / 5) |
|--------|-----------------|
| groundedness | 4.75 |
| citation_quality | 4.92 |

**Reading:** answers are almost always fully supported by the retrieved passages
and carry correct `[Art. N]` citations. Graph-route answers scored slightly lower
on groundedness (4 vs 5) — expected, since they synthesise relationships across
several articles rather than quoting one passage.

> Caveat: judge and answer model are both GPT-4o here, so scores carry some
> self-preference bias; treat them as a faithfulness smoke test, not ground truth.

## Notes & next steps

- The set is 53 hand-labelled questions; still modest, so treat numbers as a
  signal rather than a published benchmark. Add more to `eval_questions.json`
  to harden it further.
- Retrieval-only metrics don't measure answer faithfulness — see the LLM-as-judge
  results below, which score generated answers for groundedness and citation
  correctness. It is kept as a separate script so this cheap retrieval eval can
  run in CI without LLM calls.
