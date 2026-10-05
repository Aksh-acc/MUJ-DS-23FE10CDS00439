# Architecture

## The problem branching solves

The corpus holds three source types with very different shapes:

| Branch | Source | Documents | Chunks | Share of corpus | Mean chunk |
|---|---|---|---|---|---|
| `reviews` | `fancyzhx/amazon_polarity` | 2,000 | 4,557 | 26.0% | 228 chars |
| `news` | `fancyzhx/ag_news` | 2,000 | 2,045 | 11.6% | 226 chars |
| `financials` | `ashraq/financial-news-articles` | 2,000 | 10,956 | 62.4% | 501 chars |

Chunks per document: 2.28 for reviews, 1.02 for news, **5.48 for financials**.

Each branch holds the same number of documents, but financial articles are long
and chunk into five times as many retrievable units as news items. Pooled into
one index, 62% of everything retrievable is financial reporting, so a top-k
search returns whichever source is most verbose rather than whichever source
answers the question. Measured on the labelled multi-source queries, a flat
index draws **82% of its evidence from a single branch** and reaches 0.63 source
coverage, against 0.90 for the branched pipeline
([docs/results.md](results.md)).

Branching fixes this by never merging: three indexes, per-branch evidence
budgets, per-branch retrieval profiles and per-branch prompts.

## Pipeline

```
                            question
                               |
                   +-----------v-----------+
                   |   query analysis      |  normalise, keyphrases (MMR)
                   +-----------+-----------+
                               |
                   +-----------v-----------+
                   |       ROUTER          |  clause split -> per-branch score
                   |  embedding + lexical  |  relative cutoff, max 3 branches
                   +--+--------+--------+--+
                      |        |        |
         selected     |        |        |      rejected branches cost nothing
              +-------v--+  +--v-----+  +--v--------+
              | reviews  |  | news   |  | financials|
              +----+-----+  +---+----+  +-----+-----+
                   |            |             |
          per-branch query expansion from that branch's own vocabulary
                   |            |             |
             +-----v-----+ +----v------+ +----v------+
             | dense     | | dense     | | dense     |  FAISS IndexFlatIP
             |  + BM25   | |  + BM25   | |  + BM25   |  exact cosine
             +-----+-----+ +-----+-----+ +-----+-----+
                   |             |             |
                  RRF           RRF           RRF        weights per branch
                   |             |             |
              cross-encoder rerank of top 24 candidates
                   |             |             |
              top_k=6       top_k=6       top_k=5
                   |             |             |
             +-----v-----+ +-----v-----+ +-----v-----+
             | synthesis | | synthesis | | synthesis |  branch-specific prompt,
             | (reviews) | | (news)    | | (finance) |  concurrent, cited
             +-----+-----+ +-----+-----+ +-----+-----+
                   |             |             |
                   +-------------+-------------+
                                 |
                      sentiment note (reviews only)
                                 |
                      +----------v----------+
                      |      FUSION         |  merge summaries, surface
                      |  conflict + confidence |  disagreement, keep citations
                      +----------+----------+
                                 |
                      cited answer + evidence trail
```

## Stage reference

| Stage | Module | Output |
|---|---|---|
| Ingest | `data/sources.py`, `data/ingest.py` | `corpus.parquet`, `chunks.parquet` |
| Clean + chunk | `data/preprocess.py` | sentence-aligned chunks |
| Embed + index | `index/` | one FAISS index and one BM25 index per branch |
| Route | `retrieval/router.py` | `RoutingPlan` with per-branch scores |
| Expand | `retrieval/query_ops.py` | one `BranchQuery` per selected branch |
| Retrieve | `retrieval/retriever.py` | `BranchResult` with `Evidence` |
| Rerank | `retrieval/reranker.py` | reordered candidates |
| Analyse | `analysis/sentiment.py` | `SentimentReport` on reviews |
| Synthesise | `generation/synthesis.py` | `BranchSummary` per branch |
| Fuse | `generation/synthesis.py` | `FusedAnswer` with citations |
| Orchestrate | `pipeline.py` | `PipelineAnswer` with all stages + timings |

## Concurrency

Branches are independent at two stages, and both are parallelised with a thread
pool. Retrieval is dominated by native code (FAISS, BM25, torch) that releases
the GIL; branch synthesis is dominated by network wait. Both are cases where
threads give real wall-clock savings without the pickling constraints of
processes.

Citation numbers are assigned *after* all branches return, in branch order, so
they are stable regardless of which thread finished first.

## Degradation

No stage raises on failure; each reports and continues.

| Failure | Behaviour |
|---|---|
| Cross-encoder unavailable | fused order kept, precision drops |
| Sentiment model unavailable | report marked unavailable, pipeline continues |
| LLM router unparseable | falls back to the hybrid router |
| Branch synthesis fails | that branch is marked failed, others proceed |
| Fusion call fails | concatenated branch summaries shown |
| No LLM backend at all | retrieval runs, app becomes an evidence browser |
| No index built | app shows the two commands needed |

## Performance

Measured on an RTX 4060 laptop, Groq `openai/gpt-oss-120b` for generation.

| Stage | Warm |
|---|---|
| Route | 0.02 s |
| Retrieve (2 branches, hybrid + rerank) | 0.09-0.16 s |
| Keyphrases | 0.03 s |
| Branch synthesis (concurrent) | 2-3 s |
| Fusion | 1.3 s |
| **Total, retrieval only** | **< 0.2 s** |
| **Total, with generation** | **4-6 s** |

Index build: 17,558 chunks in 24 s. Corpus build: ~45 s including download.
First query of a session adds a one-off model load of roughly 8 s.
