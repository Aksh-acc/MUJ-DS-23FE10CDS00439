# Branched RAG for Market and Competitive Intelligence

A retrieval-augmented question answering system that routes a business question
to the sources capable of answering it, retrieves and summarises each source on
its own terms, and fuses the results into a single cited answer.

Built with open-weight models end to end. Runs as a Streamlit app.

---

## The idea in one example

> **"Should we launch a budget tablet? Consider demand, rivals and financial risk."**

A conventional RAG system embeds that question, searches one index, and returns
the ten nearest chunks. Because the financial articles in this corpus are long
and produce 62% of all retrievable chunks, most of those ten come back financial
regardless of the question — and the answer silently never sees a customer
review.

This system instead:

1. splits the question into clauses and discovers it has three intents
2. routes `demand` and `rivals` to market news, `financial risk` to financial
   reporting
3. searches each branch separately, with its own query, its own dense/sparse
   balance and its own evidence budget
4. summarises each branch under an instruction describing what that source can
   and cannot support
5. fuses the summaries, stating where sources agree, where they conflict, and
   how confident the result is

Measured on the labelled multi-source queries, at an identical evidence budget
with identical embeddings, BM25 and fusion:

| | branched | flat index |
|---|---|---|
| source coverage | **0.90** | 0.63 |
| distinct sources per answer | **2.20** | 1.70 |
| evidence drawn from one branch | — | 82% |

---

## Domain and data

Market and competitive intelligence over three public datasets, deliberately
chosen because their text distributions differ:

| Branch | Dataset | Documents | Chunks | Character |
|---|---|---|---|---|
| Customer Reviews | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | 2,000 | 4,557 | short, subjective, first-person |
| Market News | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | 2,000 | 2,045 | headline plus lede, topical |
| Financial Reporting | [`ashraq/financial-news-articles`](https://huggingface.co/datasets/ashraq/financial-news-articles) | 2,000 | 10,956 | long-form Reuters, entity and figure heavy |

Equal document counts, a five-fold difference in chunk counts: 5.48 chunks per
financial article against 1.02 per news item. That asymmetry is the whole reason
branching is worth its complexity, and it is measured rather than asserted.

The corpus is committed (`data/processed/`), so the app is reproducible without
re-downloading anything.

---

## Architecture

```
question
   |
   +-> query analysis ........ normalise, MMR keyphrases
   |
   +-> ROUTER ................ clause split, embedding + lexical scoring,
   |                           relative cutoff, up to 3 branches
   |
   +---------------+---------------+
   |               |               |
 reviews         news         financials        <- only selected branches run
   |               |               |
 expand          expand          expand         <- from that branch's vocabulary
   |               |               |
 dense + BM25    dense + BM25    dense + BM25   <- FAISS exact cosine
   |               |               |
  RRF             RRF             RRF           <- per-branch weights
   |               |               |
 rerank          rerank          rerank         <- cross-encoder, top 24
   |               |               |
 synthesis       synthesis       synthesis      <- branch-specific prompt
   |               |               |
   +---------------+---------------+
                   |
            sentiment note (reviews only)
                   |
                FUSION ....... merge, surface conflict, state confidence
                   |
          cited answer + full evidence trail
```

Details in [docs/architecture.md](docs/architecture.md). The reasoning behind
every choice is in [docs/design-decisions.md](docs/design-decisions.md).

---

## Models, and why each was chosen

| Role | Model | Licence | Reasoning |
|---|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | Apache-2.0 | 22M params, 384-dim. A tenth the size of mpnet at most of the retrieval quality. Indexes 17,558 chunks in 24 s, so the index can be rebuilt during a demo, and leaves memory for a reranker and generator. |
| Lexical | BM25 Okapi | — | A bi-encoder maps "EUR 47.5 million" and "EUR 52.1 million" to nearly the same point. BM25 scores rare tokens through IDF, so it is strongest exactly where the dense index is weakest. |
| Fusion | Reciprocal Rank Fusion | — | Dense cosine is bounded, BM25 is not, so any weighted score sum compares incommensurable units. RRF keeps only ranks, which are scale-free. |
| Reranking | `ms-marco-MiniLM-L-6-v2` | Apache-2.0 | Scores query and passage jointly rather than independently. Too slow for a whole branch, so applied to 24 candidates — the largest single nDCG gain measured. |
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | Apache-2.0 | Gives a countable proportion instead of an adjective. Binary because the source labels are binary. |
| Generation | `openai/gpt-oss-120b` via Groq | Apache-2.0 | Follows multi-source synthesis and citation instructions reliably, sub-second. Open weights, so it is inspectable and replaceable. |
| Generation (offline) | `qwen2.5:3b-instruct` / `Qwen2.5-1.5B-Instruct` | Apache-2.0 | Fully local alternatives so the project is never unrunnable. |

Every model is open-weight. `llm.provider: auto` picks the best available
backend and degrades to one needing no key and no daemon.

---

## Results

Full output in [docs/results.md](docs/results.md), regenerated by
`python scripts/run_eval.py`.

**Router strategies fail in opposite directions, which is why they are combined**
(exact-set match over 35 hand-labelled queries):

| strategy | single-source | multi-source | overall |
|---|---|---|---|
| embedding only | 0.560 | **0.900** | 0.657 |
| lexical only | **0.920** | 0.100 | 0.686 |
| **hybrid (default)** | 0.880 | 0.700 | **0.829** |

**Hybrid retrieval and reranking both earn their place** (known-item retrieval,
`news` branch, k=10):

| mode | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense only | 0.762 | 0.466 | 0.539 |
| sparse only | 0.675 | 0.370 | 0.443 |
| fused | 0.825 | 0.469 | 0.554 |
| **fused + rerank** | **0.875** | **0.552** | **0.631** |

Reranking improves ranking quality (MRR, nDCG) in all three branches. Recall@10
is unchanged or marginally lower, which is expected: reranking reorders a fixed
candidate pool rather than enlarging it.

Routing labels were assigned from each question's wording *before* measuring the
router, thresholds were not tuned on this set, and the six routing errors are
listed in the results rather than removed.

---

## Quick start

```bash
cd capstone
pip install -r requirements.txt

cp .env.example .env          # optional: add GROQ_API_KEY for the best backend

python scripts/build_corpus.py   # ~45 s
python scripts/build_index.py    # ~25 s
streamlit run app.py
```

Full instructions, backend options and troubleshooting:
[docs/installation.md](docs/installation.md).

---

## What the app shows

Four tabs, because a RAG demo that shows only prose is indistinguishable from a
chatbot guessing.

- **Answer** — the fused answer with inline citations, and every cited source
  resolved to its branch, dataset and link.
- **Evidence by branch** — one tab per branch: its summary, the query it was
  actually sent, the expansion terms added, and every passage with its fused
  score, rerank score, and whether dense or lexical retrieval found it.
- **Routing** — per-branch scores against the selection cutoff, with the reason
  each branch was chosen, including which clause of a compound question pulled
  it in.
- **Diagnostics** — citation grounding rate, review sentiment over the retrieved
  sample, query keyphrases, and per-stage timings.

The sidebar allows overriding the router, switching routing strategy, turning
generation off, and switching query expansion between deterministic and LLM.

---

## Layout

```
capstone/
├── app.py                      Streamlit interface
├── config.yaml                 all tunable parameters, each justified in docs
├── requirements.txt
├── .env.example
├── pytest.ini
├── src/branched_rag/
│   ├── config.py               typed config loading
│   ├── pipeline.py             BranchedRAGPipeline, the public entry point
│   ├── logging_setup.py
│   ├── data/                   dataset adapters, cleaning, chunking
│   ├── index/                  embedder, FAISS, BM25, RRF, per-branch store
│   ├── retrieval/              router, query expansion, reranker, retriever
│   ├── generation/             LLM backends, prompts, synthesis and fusion
│   ├── analysis/               sentiment aggregation
│   └── evaluation/             metrics and the evaluation harness
├── scripts/                    build_corpus, build_index, run_eval
├── tests/                      92 tests
├── data/
│   ├── processed/              committed corpus and chunks
│   └── eval/queries.jsonl      35 hand-labelled routing queries
├── docs/                       architecture, design decisions, install, results
└── screenshots/
```

---

## Known limitations

- **The `reviews` branch scores lowest on known-item retrieval** (recall@10
  0.296). Review headlines such as "Disappointed" genuinely do not identify
  their document, so the metric partly measures title quality. It remains a
  valid relative comparison across retrieval modes, which is how it is used.
- **The router over-selects `news`** on three of the labelled queries. News
  keyword overlap is broad. Over-selection costs tokens, not correctness, and
  the configuration was left untuned rather than fitted to 35 queries.
- **35 labelled queries is a small evaluation set.** Enough to separate three
  router strategies by a wide margin, not enough for fine tuning.
- **The corpus is a fixed snapshot** of three public datasets, not live market
  data. Answers describe the corpus and the UI says so.
- **Sentiment describes retrieved reviews, not the market.** Always reported
  with its sample size.
