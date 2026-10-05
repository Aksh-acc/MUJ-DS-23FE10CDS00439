# Code

The capstone source lives in [`../capstone/src/branched_rag/`](../capstone/src/branched_rag/)
as an installable package rather than being copied here, so there is one copy of
every module and the tests, notebooks and Streamlit app all import the same code.

## Package map

| Module | Responsibility |
|---|---|
| `config.py` | Typed loading of `config.yaml`; every tunable parameter in one place |
| `pipeline.py` | `BranchedRAGPipeline`, the single entry point used by the app and the evaluation harness |
| `data/sources.py` | Hugging Face dataset adapters and corpus admission rules |
| `data/preprocess.py` | Text cleaning and sentence-aware chunking |
| `data/ingest.py` | Corpus and chunk table construction |
| `index/embedder.py` | Sentence embedding with a cached model |
| `index/dense.py` | Exact cosine search over FAISS |
| `index/sparse.py` | BM25 Okapi with a domain-aware tokeniser |
| `index/fusion.py` | Reciprocal Rank Fusion |
| `index/store.py` | Per-branch index construction, persistence and loading |
| `retrieval/router.py` | Branch selection: clause splitting, embedding + lexical scoring |
| `retrieval/query_ops.py` | Keyphrase extraction and branch-conditioned query expansion |
| `retrieval/reranker.py` | Cross-encoder reranking |
| `retrieval/retriever.py` | Hybrid retrieval orchestration per branch |
| `generation/llm.py` | Groq, Ollama and local transformers backends behind one interface |
| `generation/prompts.py` | Per-branch analyst prompts and the fusion prompt |
| `generation/synthesis.py` | Branch synthesis, fusion, citation handling |
| `analysis/sentiment.py` | Sentiment aggregation over retrieved reviews |
| `evaluation/` | Metrics and the evaluation harness |

## Entry points

```bash
cd ../capstone
python scripts/build_corpus.py     # fetch, clean, chunk
python scripts/build_index.py      # embed and index
python scripts/run_eval.py         # regenerate docs/results.md
python -m pytest                   # 92 tests
streamlit run app.py               # the application
```

Design reasoning for every module is in
[`../capstone/docs/design-decisions.md`](../capstone/docs/design-decisions.md).
