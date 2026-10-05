# Capstone presentation outline

Twelve slides, roughly 10 minutes plus a live demo. Every number below is
measured and comes from `capstone/docs/results.md`.

---

## 1. Title

Branched RAG for Market and Competitive Intelligence
Name, registration number, Batch F, NLP Capstone.

---

## 2. The question

> "Should we launch a budget tablet? Consider demand, rivals and financial risk."

One question, three different kinds of evidence: what buyers say, what the
market is doing, what the numbers show. Ask it of a normal RAG system and see
what comes back.

---

## 3. Why flat RAG fails here

Three sources, equal document counts, very unequal chunk counts:

| Branch | Documents | Chunks | Chunks/doc | Share |
|---|---|---|---|---|
| Customer reviews | 2,000 | 4,557 | 2.28 | 26.0% |
| Market news | 2,000 | 2,045 | 1.02 | 11.6% |
| Financial reporting | 2,000 | 10,956 | 5.48 | 62.4% |

Financial articles are long, so they produce 62% of everything retrievable.
Pooled into one index, top-k is decided by **which source is most verbose**, not
which source answers the question.

Speaker note: this is the whole motivation. Land it before showing architecture.

---

## 4. The architecture

The pipeline diagram from `capstone/README.md`.

Route by clause → retrieve each branch separately (own query, own dense/sparse
balance, own budget) → summarise each under its own prompt → fuse, surfacing
disagreement.

---

## 5. Routing, and why it is hybrid

Exact-set match over 35 hand-labelled queries:

| strategy | single-source | multi-source | overall |
|---|---|---|---|
| embedding only | 0.560 | **0.900** | 0.657 |
| lexical only | **0.920** | 0.100 | 0.686 |
| **hybrid** | 0.880 | 0.700 | **0.829** |

The two signals fail in **opposite** directions. Embedding generalises to
paraphrase but over-selects; keywords are precise but blind to paraphrase and
collapse to 0.100 on compound questions. That is the argument for combining
them, and it is measured rather than assumed.

---

## 6. Clause splitting: a bug and its fix

The flagship question originally routed to `news` only — the financial half was
silently dropped, because one vector for three intents lands near none of them.

Splitting into clauses and taking each branch's best clause score raises
`financials` from **0.117 to 0.501** and the branch is correctly selected.

Speaker note: good slide for "innovation and problem solving". It was found by
testing, not by design.

---

## 7. Hybrid retrieval

Dense misses figures: a bi-encoder maps "EUR 47.5 million" and "EUR 52.1
million" to nearly the same point. BM25 scores rare tokens through IDF.

RRF fuses them by **rank**, not score, because dense cosine is bounded and BM25
is not — summing them would compare incommensurable units.

| mode (news branch) | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| dense only | 0.762 | 0.466 | 0.539 |
| sparse only | 0.675 | 0.370 | 0.443 |
| fused | 0.825 | 0.469 | 0.554 |
| **fused + rerank** | **0.875** | **0.552** | **0.631** |

Be precise: reranking improves *ordering* in all three branches; recall is
unchanged or slightly lower, because it reorders a fixed pool.

---

## 8. The headline result: branched vs flat

Identical embeddings, identical BM25, identical RRF, identical evidence budget.
Only the partitioning differs.

| multi-source queries | branched | flat |
|---|---|---|
| source coverage | **0.90** | 0.63 |
| distinct sources per answer | **2.20** | 1.70 |
| evidence from one branch | — | **82%** |

The flat index concentrates 82% of its evidence in a single branch. This is the
slide to spend time on.

---

## 9. Data quality work

Three defects found by inspecting data, not by reasoning:

1. `ag_news` encodes spaces as backslashes (`dwindling\band`). Decoding `\b` as
   an escape deletes a word.
2. The review corpus came out **67% negative** because the admission filter used
   valence-carrying cues — it was selecting complaints, not reviews. Fixed with
   neutral cues plus balanced sampling (1000/1000).
3. **7.1% of financial chunks exceeded their budget**, the largest at 13,066
   characters, from contact blocks with no sentence boundary. A word-boundary
   ceiling brought the maximum to 717.

Result: 0 of 17,558 chunks retain an artifact, asserted as a test.

---

## 10. Open-weight models throughout

| Role | Model | Licence |
|---|---|---|
| Embeddings | all-MiniLM-L6-v2 | Apache-2.0 |
| Reranking | ms-marco-MiniLM-L-6-v2 | Apache-2.0 |
| Sentiment | distilbert SST-2 | Apache-2.0 |
| Generation | openai/gpt-oss-120b (Groq) | Apache-2.0 |
| Generation, offline | Qwen2.5-1.5B / 3B-Instruct | Apache-2.0 |

Nothing depends on a closed model. `provider: auto` degrades from Groq to Ollama
to local transformers, so the demo runs with no key at all.

---

## 11. Live demo

Order matters:

1. Run the tablet question. Show the **Routing** tab first — scores, cutoff, and
   which clause pulled in each branch.
2. **Evidence by branch** — the query each branch actually received, expansion
   terms, and `dense + lexical` versus `dense only` per passage.
3. **Answer** — citations resolving to real sources, plus the Agreement and
   Confidence lines.
4. **Diagnostics** — 100% citation grounding, sentiment over the retrieved
   sample, stage timings.
5. Force a branch the router rejected, to show the override.

Fallback: `docs/results.md` and the screenshots, in case of no network.

---

## 12. Limitations and what is next

Stated honestly:

- Reviews branch scores lowest on known-item retrieval (0.296) — review titles
  like "Disappointed" genuinely do not identify a document.
- The router over-selects `news` on three labelled queries; thresholds were
  deliberately **not** tuned on a 35-query set.
- Fixed corpus snapshot, not live market data.

Next: a larger labelled set, per-branch threshold tuning with proper
train/test separation, and a cross-encoder reranker fine-tuned per branch.

---

## Preparing the deck

Build from this outline, then capture screenshots into
`capstone/screenshots/`:

| File | What to capture |
|---|---|
| `01_app_overview.png` | Landing page with the sidebar backend status visible |
| `02_answer_with_citations.png` | Answer tab, citations and cited sources resolved |
| `03_evidence_by_branch.png` | A branch tab with an expanded passage showing scores |
| `04_routing.png` | Routing tab for the compound tablet question |
| `05_diagnostics.png` | Diagnostics tab: grounding, sentiment, timings |

```bash
cd capstone
streamlit run app.py
```
