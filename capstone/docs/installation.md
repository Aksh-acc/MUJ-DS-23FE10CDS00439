# Installation and running

## Requirements

- Python 3.10 or newer (developed on 3.13)
- About 3 GB of disk for model weights and indexes
- Internet access for the first run, to download datasets and models
- A GPU is optional. It speeds up indexing (24 s versus roughly 3 minutes) but
  nothing requires it.

## 1. Install

```bash
cd capstone
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Configure a generation backend

Retrieval works with no configuration at all. Generation needs one of three
backends, and `config.yaml` ships with `llm.provider: auto`, which tries them in
order and uses the first that is available.

**Option A - Groq (recommended).** A free key serves `openai/gpt-oss-120b`,
Apache-2.0 open weights, in well under a second.

```bash
cp .env.example .env
# then put your key in .env:
# GROQ_API_KEY=gsk_...
```

Get a key at <https://console.groq.com/keys>.

**Option B - Ollama (fully offline).**

```bash
ollama pull qwen2.5:3b-instruct
ollama serve
```

**Option C - nothing at all.** `auto` falls through to
`Qwen/Qwen2.5-1.5B-Instruct` via transformers. No key and no daemon; the first
run downloads about 3 GB.

`.env` is listed in `.gitignore`. Never commit it.

## 3. Build the corpus and index

```bash
python scripts/build_corpus.py    # downloads, cleans, chunks  (~45 s)
python scripts/build_index.py     # embeds and indexes         (~25 s on GPU)
```

Expected output:

```
    branch  documents  chunks  mean_chunk_chars
   reviews       2000    4557             227.7
      news       2000    2045             225.8
financials       2000   10956             500.5

17558 chunks indexed
```

`corpus.parquet` and `chunks.parquet` are committed, so `--chunks-only` can
re-chunk without re-downloading if you change chunk geometry:

```bash
python scripts/build_corpus.py --chunks-only
```

## 4. Run the app

```bash
streamlit run app.py
```

Opens on <http://localhost:8501>.

## 5. Optional: tests and evaluation

```bash
python -m pytest                  # 92 tests, about 50 s
python scripts/run_eval.py        # regenerates docs/results.md
python scripts/run_eval.py --sample 40   # faster
```

The test suite needs no API key: integration tests run with generation off, and
unit tests use a fake embedder.

## Troubleshooting

**"no index in .../artifacts"** — run the two build scripts from step 3. The app
shows this message with the commands.

**"index was built with X but config requests Y"** — the embedding model in
`config.yaml` changed. Re-run `build_index.py`.

**Dataset download fails or times out** — the loaders stream from Hugging Face
and retry automatically. A transient `WinError 10038` during download is
retried; if it persists, re-run `build_corpus.py`, which restarts cleanly.

**First query is slow (about 8 s)** — one-off load of the embedder, cross-encoder
and sentiment model. Later queries are well under a second for retrieval.

**No backend available** — the sidebar says so and the app runs as an evidence
browser. Retrieval, routing and the evidence trail all still work.

**Out of memory on GPU** — set `llm.provider: groq` so no local generator
loads, or switch `local_hf.model` to `Qwen/Qwen2.5-0.5B-Instruct`.
