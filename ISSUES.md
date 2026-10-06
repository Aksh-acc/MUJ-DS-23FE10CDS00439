# Issue log

Issues raised and worked during the capstone. Each entry is written so it can be
pasted straight into GitHub Issues: title, labels, body, and the commit or
approach that closed it.

The first twelve are **real defects found while building**, most of them by
inspecting data or output rather than by reasoning about the code. The remaining
five are the planning issues covering the rubric's required categories.

---

## Resolved

### #1 `ag_news` encodes inter-word spaces as backslashes, corrupting tokens
`labels: bug, data-quality`

**Problem.** Raw `ag_news` rows arrive with single backslashes where spaces
should be: `Wall Street's dwindling\band of ultra-cynics`. 537 of the first 4,000
rows are affected.

**Why it matters.** The obvious fix — treating `\b` as a C-style escape —
produces "dwindling and" and silently *deletes a word*. Left alone, the
backslash fuses two words into one out-of-vocabulary token and both BM25 and the
embedder lose both terms.

**Resolution.** Unescape genuinely escaped characters first (`\$`, `\"`), then
restore any surviving lone backslash to a space. Order matters and is asserted
by `test_lone_backslash_becomes_a_space`.

---

### #2 HTML entities arrive with the leading ampersand stripped
`labels: bug, data-quality`

**Problem.** Scraped rows contain `#36;46` where "$46" belongs, and `isn#39;t`
for "isn't". `html.unescape` does not touch these because the `&` is missing.

**Why it matters.** The digits survive into the index as nonsense tokens and
figures become unsearchable.

**Resolution.** Repair the fragments to `&#36;` / `&#39;` before unescaping.
Verified by a corpus-wide assertion that 0 of 17,558 chunks retain an artifact.

---

### #3 Sentence splitter tore decimals and abbreviations in half
`labels: bug, nlp`

**Problem.** A naive split on `.!?` cuts "EUR 47.5 million" and "The U.S. unit"
into two sentences. Both forms are dense in the financials branch.

**Why it matters.** A chunk boundary inside a figure destroys the figure, which
is exactly what financial questions turn on.

**Resolution.** A fixed-width regex lookbehind cannot express "unless the
preceding token is an abbreviation" — Python raises
`look-behind requires fixed-width pattern`. Candidate boundaries are generated
with `finditer` and then filtered against an abbreviation set, an initials
check, and a trailing-digit check.

---

### #4 Product-cue filter matched substrings and admitted a film review
`labels: bug, data-quality`

**Problem.** The reviews branch contained "A romantic zen baseball comedy".
The cue `motor` had substring-matched *motorcycle*.

**Resolution.** Cues became a compiled alternation with `\b` word boundaries,
the requirement rose to two distinct cues, and the media exclusion list was
widened to cover records, vinyl and compilations.

---

### #5 Review corpus was 67% negative, biasing every sentiment aggregate
`labels: bug, data-quality, high-priority`

**Problem.** After #4 was fixed, the reviews branch was still 1,343 negative to
657 positive.

**Why it matters.** The cue list contained valence-carrying terms —
`defective`, `flimsy`, `overpriced`, `malfunction`. The filter was selecting
*complaints*, not *product reviews*. Any sentiment aggregate computed over that
corpus would measure corpus construction rather than the market, and a
market-research answer would have wrongly concluded products are hated.

**Resolution.** Two changes. Cues reduced to product nouns and neutral commerce
terms only, and the sample balanced to exactly 1,000 positive / 1,000 negative
at ingest. Asserted by `test_reviews_branch_is_polarity_balanced`.

---

### #6 Router threshold was calibrated for the wrong score range
`labels: bug, retrieval`

**Problem.** `min_score: 0.34` was set by intuition. Descriptor cosine
similarity for `all-MiniLM-L6-v2` actually spans roughly 0.0-0.45, so almost
every query fell through to the "keep the top branch anyway" fallback and
multi-branch selection effectively never fired.

**Resolution.** Selection became relative: keep a branch scoring within 55% of
the best branch, with `min_score` demoted to an absolute floor of 0.08. The
right question is which branches are *competitive for this query*, which is
inherently relative.

---

### #7 Compound questions routed to a single branch
`labels: bug, retrieval, high-priority`

**Problem.** The project's flagship query — *"Should we launch a budget tablet?
Consider demand, rivals and financial risk"* — selected only `news`. The
financial half of the question was silently dropped.

**Why it matters.** Encoding three intents as one vector lands near none of
them. This defeated the whole point of branching on exactly the query the
system exists to answer.

**Resolution.** Split the query on sentence punctuation and coordinators, score
every clause against every branch, and give each branch its best clause score.
The clause `financial risk` raises the financials score from **0.117 to 0.501**
and the branch is now correctly selected. Regression-tested by
`test_compound_question_selects_several_branches`.

---

### #8 Query expansion wasted slots on morphological duplicates
`labels: bug, retrieval`

**Problem.** Branch keyword lists carry singular and plural forms which embed
almost identically, so the top-3 expansion returned `costs`, `profitability`,
`cost` — two spellings of one concept.

**Resolution.** Skip near-duplicates by prefix before accepting a term. A
stemmer would be overkill for comparing short config-declared keywords.

---

### #9 `gpt-oss` emits non-ASCII citation markers, zeroing the grounding metric
`labels: bug, generation, high-priority`

**Problem.** The model intermittently returns citations as `【5†L0-L2】` instead
of `[5]` — it is trained with a browsing tool and falls back to that tool's
syntax. Observed **non-deterministically on identical input**.

**Why it matters.** The citation parser returned an empty set, so the grounding
metric silently read zero while the answer on screen looked perfectly cited.
A metric that fails quietly is worse than no metric.

**Resolution.** Normalise all marker variants to ASCII `[n]` before parsing or
display, *and* forbid the form in the prompt. The prompt alone cannot be relied
on for behaviour that varies run to run. Covered by
`test_full_width_browser_markers_are_normalised`.

---

### #10 Chunks up to 13,066 characters, twenty times their budget
`labels: bug, data-quality, high-priority`

**Problem.** Auditing chunk lengths showed 7.1% of financial chunks over budget,
the largest at 13,066 characters against a 640 limit.

**Cause.** Press-release contact blocks contain no usable sentence boundary at
all — every period sits inside an email address or follows an abbreviation
(`Ms. Cara O'Brien ... cara.obrien@fticonsulting.com Tel: +852-...`), so the
splitter correctly found no boundary and emitted the whole block as one
sentence.

**Why it matters.** A 13k-character span compressed into a 384-dimension vector
is semantic mush, and the cross-encoder silently truncates at 512 tokens, so the
rerank score described only the opening of the chunk.

**Resolution.** A word-boundary ceiling for spans that defeat sentence
segmentation. Over-budget financial chunks fell **725 to 80** and the maximum
**13,066 to 717**. Single tokens longer than the budget (URLs, identifiers) are
deliberately left intact.

---

### #11 Configured Groq model not available on the account
`labels: bug, infrastructure`

**Problem.** `llama-3.3-70b-versatile` returned
`404 model_not_found`. Listing the account's models showed a different roster.

**Resolution.** Switched the default to `openai/gpt-oss-120b` (Apache-2.0 open
weights, benchmarked at 0.49 s) with `openai/gpt-oss-20b` as fallback, and made
the provider retry the fallback automatically. Also handles the case where a
reasoning model spends its whole token budget thinking and returns empty
content.

---

### #12 Two environment papercuts
`labels: bug, infrastructure`

- `python-dotenv`'s `find_dotenv()` raises `AssertionError` when called from
  stdin, because it walks the caller's stack frames. Fixed by always passing an
  explicit path to `load_dotenv`.
- Streamlit 1.65 deprecates `use_container_width`. Replaced with
  `width="stretch"` across the app; the test harness now runs warning-free.
- `takala/financial_phrasebank` and `McAuley-Lab/Amazon-Reviews-2023` are
  script-based datasets, which current `datasets` refuses to load. Both were
  rejected during source selection in favour of parquet-backed alternatives.

---

## Planned work items

### #13 Dataset collection
`labels: data, enhancement`

Expand beyond the fixed 2,000-document-per-branch snapshot. Document any new
source's licence, add an adapter to `data/sources.py`, and re-verify the
cleaning assertions, since every cleaning rule so far was driven by a defect
specific to a source.

### #14 Model development
`labels: model, enhancement`

Fine-tune a cross-encoder per branch rather than using one MS MARCO checkpoint
for all three. The branches have very different text distributions, and the
`reviews` branch is where reranking helps least.

### #15 Testing
`labels: testing`

Current suite is 92 tests. Gaps worth closing: the Ollama and local
transformers backends are exercised only through their availability checks, and
there is no test for the LLM router path.

### #16 Documentation
`labels: documentation`

Capture the five app screenshots listed in `capstone/screenshots/README.md` and
build the slide deck from `presentations/capstone_presentation_outline.md`.

### #17 Deployment
`labels: deployment, enhancement`

Deploy to Streamlit Community Cloud. Blockers: committed index artifacts exceed
comfortable repository limits, so the index must either be built on first boot
or fetched from release assets; `GROQ_API_KEY` must move to Streamlit secrets.
