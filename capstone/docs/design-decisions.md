# Design decisions

Every non-obvious choice, with the reason and the alternative rejected. Where a
decision was driven by something measured, the measurement is given.

---

## 1. Branched RAG rather than flat RAG

**Decision.** Three separate indexes, a router that selects among them, one
synthesis per branch, then a fusion step.

**Why.** The three sources contribute wildly unequal numbers of chunks from
equal numbers of documents: financial articles are long and produce 62% of all
retrievable units (5.48 chunks per document), news items are short and produce
12% (1.02 chunks per document). In a single pooled
index, top-k is decided largely by which source is most verbose.

Measured on the labelled multi-source queries, at an identical evidence budget
and with identical embeddings, BM25 and RRF:

| | branched | flat |
|---|---|---|
| source coverage | **0.90** | 0.63 |
| distinct sources per answer | **2.20** | 1.70 |
| evidence from single largest branch | - | 82% |

**Also bought.** Per-branch retrieval profiles (a pooled index has one global
dense/sparse weight), per-branch prompts, and the ability to skip a branch
entirely. **Cost.** More moving parts, a router that can be wrong, and N+1 LLM
calls instead of 1.

**Rejected.** Flat RAG with a larger k — raising k dilutes precision and still
does not guarantee any news chunk appears. Metadata filtering on one index —
expresses the budget but not the per-branch weights or prompts.

---

## 2. Hybrid router, with embedding and lexical signals combined

**Decision.** Score each branch as `0.65 * descriptor_similarity + 0.35 *
keyword_overlap`; keep branches within 55% of the best score.

**Why.** The two signals fail in opposite directions, which is measurable:

| strategy | exact match, single-source | exact match, multi-source | overall |
|---|---|---|---|
| embedding only | 0.560 | **0.900** | 0.657 |
| lexical only | **0.920** | 0.100 | 0.686 |
| **hybrid** | 0.880 | 0.700 | **0.829** |

Embedding similarity generalises to paraphrase but over-selects on narrow
questions. Keyword overlap is precise on domain terms and blind to paraphrase,
collapsing to 0.100 on compound questions. Combining them beats both.

**Rejected.** An LLM router (implemented, selectable, not default): it adds a
network round trip before retrieval starts, cannot be evaluated offline, and is
non-deterministic, so the UI could not show a stable reason for a branch
choice. Reported for comparison rather than used.

---

## 3. Clause-level routing for compound questions

**Decision.** Split the query on sentence punctuation and coordinators, score
every clause against every branch, and give each branch its best clause score.

**Why.** Found by testing, not by design. *"Should we launch a budget tablet?
Consider demand, rivals and financial risk"* encodes as a single vector near
none of its three intents, and the first router selected only `news`, silently
dropping the financial half of the question. After clause splitting, the clause
`"financial risk"` raises the financials score from 0.117 to 0.501 and the
branch is correctly selected.

One clause needing a branch is sufficient to select it, which is the right
semantics for a compound question.

**Rejected.** A dependency parse would split more accurately but needs a parser
model; the failure mode of the crude split is benign, since an over-split
fragment simply contributes a weaker score to a maximum.

---

## 4. Relative branch-selection threshold

**Decision.** Keep a branch if `score >= 0.55 * best_score`, with `min_score =
0.08` as an absolute floor only.

**Why.** A fixed cutoff cannot work. Descriptor cosine similarity for this
encoder spans roughly 0.0-0.45, and the usable range shifts with query length,
so the first version's `min_score = 0.34` caused nearly every query to fall
through to the "keep the top branch" fallback. The question that matters is
which branches are *competitive for this query*, which is inherently relative.

---

## 5. Hybrid retrieval: dense + BM25 fused by RRF

**Decision.** Run both retrievers per branch and fuse by Reciprocal Rank Fusion
with per-branch weights.

**Why dense alone is insufficient.** A 22M-parameter bi-encoder maps "EUR 47.5
million" and "EUR 52.1 million" to nearly the same point — it encodes *European
currency amount* and discards which one. Tickers, model numbers and company
names are exactly the tokens these questions turn on, and exactly what BM25
scores well through IDF.

**Why RRF rather than a weighted score sum.** Dense cosine sits in [0, 1]; BM25
is unbounded and its scale depends on corpus statistics, so the same raw score
means different things in different branches. Summing them compares
incommensurable units, and per-query min-max normalisation makes the weights
depend on how good the best hit happened to be. RRF discards scores and keeps
ranks, which are scale-free, so the configured weights mean the same thing
everywhere. `k=60` is the published default from the original paper, used
because it is the default and not because it was tuned here.

**Measured.** Fusion beats both single retrievers on recall in every branch,
and reranking then adds the largest gain:

| branch | mode | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| news | dense | 0.762 | 0.466 | 0.539 |
| news | sparse | 0.675 | 0.370 | 0.443 |
| news | fused | 0.825 | 0.469 | 0.554 |
| news | **fused+rerank** | **0.875** | **0.552** | **0.631** |

Reranking's effect is stated precisely: it improves **ranking quality** (MRR and
nDCG) in all three branches, while recall@10 is unchanged or marginally lower.
That is the expected behaviour, not a defect -- reranking reorders a fixed
candidate pool rather than enlarging it, so it can move a relevant item from
rank 11 to rank 3 but cannot add one that fusion never retrieved.

---

## 6. Per-branch dense/sparse weights

| branch | dense | sparse | reason |
|---|---|---|---|
| `reviews` | 0.7 | 0.3 | informal paraphrase-heavy prose; few rare keys |
| `news` | 0.6 | 0.4 | headline vocabulary, named companies |
| `financials` | 0.5 | 0.5 | dense with figures, tickers and currency amounts that need literal matching |

A pooled index cannot express this at all, which is part of decision 1.

---

## 7. Per-branch chunk geometry

| branch | chunk | overlap | reason |
|---|---|---|---|
| `reviews` | 320 | 40 | a review is usually one complete opinion; larger windows merge unrelated reviews |
| `news` | 480 | 60 | headline plus lede is self-contained |
| `financials` | 640 | 80 | long articles where a figure and its period are often sentences apart |

**Sentence-aware packing, not fixed stride.** A chunk cut mid-clause gives the
embedder a truncated proposition, and evidence shown in the UI has to be
quotable. Overlap is satisfied by repeating whole trailing sentences, so no
chunk ever begins mid-sentence.

---

## 7a. A hard ceiling on spans that defeat sentence segmentation

**Decision.** A span longer than `chunk_size` on its own is split at word
boundaries.

**Why.** Found by auditing chunk lengths rather than by reasoning. Press-release
contact blocks contain no usable sentence boundary at all -- every period sits
inside an email address or follows an abbreviation ("Ms. Cara O'Brien ...
cara.obrien@fticonsulting.com Tel: +852-...") -- so the splitter correctly finds
no boundary and emits the whole block as one sentence.

Measured before the guard: **7.1% of financial chunks exceeded their budget and
the largest reached 13,066 characters**, twenty times the limit. This is
actively harmful, not untidy. A 13k-character span compressed into a
384-dimension vector is semantic mush, and the cross-encoder silently truncates
at 512 tokens, so its rerank score described only the opening of the chunk.

After the guard, financial over-budget chunks fell from 725 to 80 and the
maximum from 13,066 to 717 characters. The residue is single tokens longer than
the budget -- URLs and identifiers -- which are deliberately left intact, since
truncating them destroys the identifier without making the chunk usable.

---

## 8. Embedding model: `all-MiniLM-L6-v2`

22M parameters, 384 dimensions. Roughly a tenth the size of `all-mpnet-base-v2`
while keeping most of its retrieval quality on short passages. The deciding
factors were concrete: the index is rebuilt during marking and demos, so
encoding 17,558 chunks has to finish in minutes (it takes 24 s on an RTX 4060,
about 3 minutes on CPU), and 384-dimension vectors leave room to hold a
reranker and a generator in memory at the same time.

Vectors are L2-normalised at encode time so a FAISS inner-product index computes
exact cosine similarity.

---

## 9. FAISS `IndexFlatIP`, not an approximate index

Brute-force exact search. Each branch holds 2k-10k vectors, where a full scan is
already sub-millisecond. An IVF or HNSW index would add a training step, tuning
parameters and a recall ceiling in exchange for nothing measurable at this
scale. The `DenseIndex` wrapper isolates the choice, so swapping in an
approximate index later touches one file.

---

## 10. Cross-encoder reranking of the top 24 only

A bi-encoder embeds query and passage independently, so neither ever sees the
other — fast enough to index 16k chunks, blunt at the top of the ranking where
all survivors are on-topic. A cross-encoder scores the pair jointly and is far
more accurate at that distinction, and far too slow for a whole branch. Applying
it to 24 candidates bounds cost by configuration rather than corpus size. The
measured nDCG gain is the largest single improvement in the retrieval table.

---

## 11. Branch-conditioned query expansion from declared vocabulary

**Decision.** For each selected branch, append the branch-vocabulary terms
closest to the query in embedding space, skipping morphological duplicates.

**Why.** The three corpora use different words for the same concept: battery
dissatisfaction is "died after two months" in a review and "handset margin
pressure" in a financial report. Sending one identical string to all three
indexes throws away the structure branching created.

Expansion draws only from the branch's keyword list in `config.yaml`, so a term
can be added only if declared relevant. That keeps it auditable — the UI shows
which terms were added — and costs no LLM round trip. An LLM rewrite is
available and falls back to this on any error.

**Fixed during development.** Keyword lists carry singular and plural forms
which embed almost identically, so the top-3 was spending slots on `cost` and
`costs`. Near-duplicates are now skipped by prefix.

---

## 12. Prompt specialisation per branch

Having split retrieval by source, one generic "answer from the context" prompt
throws the split away. Each branch gets an analyst instruction stating what that
source **can** and **cannot** support:

- reviews can establish what users report, never market size
- news can establish that something was reported and when, never its magnitude
- financials can establish stated figures, never why customers churn

Those negative constraints are explicit because the common failure here is not
fabrication but **overreach** — answering a market-demand question from three
reviews. The fusion prompt then receives summaries rather than raw chunks, and
is instructed to surface disagreement rather than average it, because reviews
and financials pointing opposite ways is a finding, not noise.

---

## 13. Fusion consumes summaries, not raw chunks

Raw chunks from three branches overflow a useful context window and bias the
model toward whichever source is longest — reintroducing precisely the volume
bias that separate indexes were built to remove. A summary already constrained
to one source's competence is also easier to attribute, which is what lets the
fusion step say "reviews support this, financials do not".

When only one branch is selected, the fusion call is skipped: paying for a
second round trip to restate a single summary buys nothing.

---

## 14. Open-weight models throughout, with three interchangeable backends

| provider | model | licence | role |
|---|---|---|---|
| `groq` | `openai/gpt-oss-120b` | Apache-2.0 | default; strongest at multi-source synthesis, sub-second |
| `ollama` | `qwen2.5:3b-instruct` | Apache-2.0 | fully offline and private |
| `local_hf` | `Qwen2.5-1.5B-Instruct` | Apache-2.0 | zero-configuration fallback, no key or daemon |

Every weight set is publicly downloadable, so no part of the system depends on a
closed model that cannot be inspected or replaced. `provider: auto` resolves in
the order above, preferring quality when a key is present and degrading to
something that always runs. The abstraction is one `complete` call, which is the
entire surface the pipeline needs and a thin enough seam for the test suite to
stub.

---

## 15. Citation marker normalisation

**Decision.** Rewrite non-ASCII citation markers to `[n]` before parsing or
display.

**Why.** Found by testing. `gpt-oss` is trained with a browsing tool and
intermittently falls back to that tool's syntax, emitting `【5†L0-L2】` instead of
`[5]` — non-deterministically, on identical input. Unhandled, the citation
parser returned an empty set and the grounding metric silently read zero while
the answer looked perfect. The prompt now forbids the form *and* the output is
normalised, because a prompt rule cannot be relied on for something observed to
vary run to run.

---

## 16. Sentiment measured, and deliberately limited

A language model asked how customers feel returns a confident adjective and
nothing countable. `distilbert-base-uncased-finetuned-sst-2-english` scores each
retrieved review chunk so the answer can cite a proportion.

Binary, because the source dataset's own labels are binary and a five-point
prediction would imply precision the data does not have. **Reviews branch only**
— SST-2 applied to a Reuters earnings report returns a confident label for text
carrying no sentiment, which measures nothing. The aggregate is always reported
with its sample size and described as the retrieved sample, never the market.

---

## 17. Corpus admission rules, and a bias found and fixed

`amazon_polarity` mixes physical goods with books, music and film. Market
intelligence concerns things that get manufactured and shipped, so a review is
admitted only on **two distinct product cues matched at word boundaries**.

Two defects were found by inspecting the output rather than by reasoning:

1. **Substring matching admitted a film review**, because `motor` occurs inside
   *motorcycle*. Cues are now a compiled alternation with `\b` boundaries.
2. **The corpus came out 67% negative.** The cue list contained valence-carrying
   terms (`defective`, `flimsy`, `overpriced`), so the filter was selecting
   *complaints* rather than *product reviews* — which would have biased every
   sentiment aggregate the system produces. Cues are now product nouns and
   neutral commerce terms only, and the sample is additionally balanced to
   exactly 1000 positive / 1000 negative at ingest.

The filter stays lexical and visible rather than becoming a learned classifier,
because corpus admission has to be defensible in a report. Precision is nearly
free here: 2,000 slots are filled from 3.6M candidates, so recall is irrelevant.

---

## 18. Cleaning rules driven by measured defects

Each rule exists because of a defect found in the data, and the **order
matters**:

1. `ag_news` encodes inter-word spaces as single backslashes
   (`dwindling\band`). Decoding `\b` as an escape yields "dwindling and" and
   silently deletes a word. Genuine escapes (`\$`, `\"`) are unescaped first,
   then a surviving lone backslash becomes a space.
2. HTML entities lost their leading ampersand during scraping, leaving `#36;46`
   for "$46" and `isn#39;t` for "isn't". Repaired before unescaping, or the
   digits enter the index as nonsense tokens.
3. Abbreviations and decimals must not be treated as sentence ends, or
   "EUR 47.5 million" and "U.S. unit" are torn in half — both dense in the
   financials branch. A fixed-width regex lookbehind cannot express the
   exception, so candidate boundaries are generated and then filtered.

Verified: **0 of 17,558 chunks** retain an escape or entity artifact, asserted
as a test so a source-format change is caught as a corpus bug.

---

## 19. Known-item retrieval evaluation

Relevance judgements for 17,558 chunks are not obtainable for a student project,
and self-graded LLM relevance would be circular. Instead a document **title** is
the query and that document's own chunks are the relevant set: if asking for a
document by its headline does not return it, retrieval is broken. No human
labels, so it reproduces exactly.

Generic placeholder titles ("Market report") are excluded because they identify
nothing, and sampling is one query per *document* so long documents do not
dominate.

The `reviews` branch scores lowest (recall@10 0.296) and this is reported rather
than hidden: review headlines like "Disappointed" genuinely do not identify
their document, so the metric is partly measuring title quality. It is a valid
*relative* comparison across retrieval modes within a branch, which is what it
is used for.

---

## 20. Routing labels assigned independently of the implementation

`data/eval/queries.jsonl` records which sources each question *should* consult,
judged from its wording before measuring what the router does, so the numbers
can and do disagree with the implementation — six routing errors are listed in
[results.md](results.md) rather than removed.

Thresholds were set a priori and **not** tuned against this set. With 35
queries, tuning on it and reporting the result as validation would be
meaningless, so the configuration is left where reasoning put it and the
measured accuracy is reported as found.

---

## 21. Rebuilding BM25 at load instead of pickling it

Tokenising 17,558 short chunks takes under a second, and a pickled `BM25Okapi`
breaks silently whenever `rank_bm25` or the tokeniser changes. The saved
artifacts are the FAISS index and the chunk table; the sparse index is derived.

A short stop list is used on purpose: aggressive stopping removes "no", "not"
and "down", which carry the signal in a complaint or a profit warning, and
BM25's IDF term already discounts anything appearing everywhere. Internal `'`,
`.`, `$`, `%` and `-` are kept inside tokens so `q3`, `47.5`, `$55.8bn` and `8%`
survive as single units.

---

## 22. Everything degrades, nothing raises

A demo that dies on a cold cache or an expired key is worth less than one that
says what it could not do. Full table in
[architecture.md](architecture.md#degradation).
