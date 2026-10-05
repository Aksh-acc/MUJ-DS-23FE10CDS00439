# References

Sources behind the design decisions, grouped by the component they informed.

## Retrieval-augmented generation

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020). <https://arxiv.org/abs/2005.11401>
  The original RAG formulation this project extends by routing and branching.
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A
  Survey* (2023). <https://arxiv.org/abs/2312.10997>
  Taxonomy of naive, advanced and modular RAG; the branch-then-fuse pattern used
  here sits in the modular category.
- Asai et al., *Self-RAG: Learning to Retrieve, Generate and Critique through
  Self-Reflection* (2023). <https://arxiv.org/abs/2310.11511>
  Background on deciding *whether* and *what* to retrieve, which is the question
  the router answers.

## Rank fusion and hybrid retrieval

- Cormack, Clarke and Buettcher, *Reciprocal Rank Fusion Outperforms Condorcet
  and Individual Rank Learning Methods* (SIGIR 2009).
  <https://dl.acm.org/doi/10.1145/1571941.1572114>
  Source of the RRF formula and of `k=60`, used as the published default rather
  than tuned on this corpus.
- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009). <https://dl.acm.org/doi/10.1561/1500000019>
  BM25 and the IDF weighting that makes it strong on rare tokens such as
  tickers and currency amounts.

## Embeddings and reranking

- Reimers and Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks* (EMNLP 2019). <https://arxiv.org/abs/1908.10084>
  The bi-encoder versus cross-encoder distinction that motivates two-stage
  retrieval.
- Nogueira and Cho, *Passage Re-ranking with BERT* (2019).
  <https://arxiv.org/abs/1901.04085>
  Cross-encoder reranking of a shortlist, the pattern used here on 24
  candidates per branch.
- Wang et al., *MiniLM: Deep Self-Attention Distillation* (2020).
  <https://arxiv.org/abs/2002.10957>
  The distillation behind both `all-MiniLM-L6-v2` and
  `ms-marco-MiniLM-L-6-v2`.
- Model cards:
  - <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
  - <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2>
  - <https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english>

## Keyphrase extraction

- Carbonell and Goldstein, *The Use of MMR, Diversity-Based Reranking for
  Reordering Documents* (SIGIR 1998).
  <https://dl.acm.org/doi/10.1145/290941.291025>
  Maximal Marginal Relevance, used to stop the top keyphrases being
  near-duplicates of each other.
- Grootendorst, *KeyBERT* (2020). <https://maartengr.github.io/KeyBERT/>
  The embed-the-document, embed-the-candidates formulation reimplemented in
  `retrieval/query_ops.py` against the existing retrieval embedder.

## Vector search

- Johnson, Douze and Jégou, *Billion-scale similarity search with GPUs* (2017).
  <https://arxiv.org/abs/1702.08734>
  FAISS. Relevant here mainly for what is *not* used: the approximate indexes
  that only pay off well above this corpus size.

## Models and infrastructure

- OpenAI, *gpt-oss* model card (2025). <https://huggingface.co/openai/gpt-oss-120b>
  Apache-2.0 open-weight mixture-of-experts model used as the default generator.
- Qwen Team, *Qwen2.5 Technical Report* (2024).
  <https://arxiv.org/abs/2412.15115>
  The local and Ollama fallback generators.
- Groq API documentation. <https://console.groq.com/docs>
- Streamlit documentation. <https://docs.streamlit.io>

## Datasets

- `fancyzhx/amazon_polarity` — Zhang, Zhao and LeCun, *Character-level
  Convolutional Networks for Text Classification* (2015).
  <https://huggingface.co/datasets/fancyzhx/amazon_polarity>
- `fancyzhx/ag_news` — AG's news corpus, same paper.
  <https://huggingface.co/datasets/fancyzhx/ag_news>
- `ashraq/financial-news-articles` — Reuters financial articles.
  <https://huggingface.co/datasets/ashraq/financial-news-articles>

## Evaluation methodology

- Järvelin and Kekäläinen, *Cumulated Gain-Based Evaluation of IR Techniques*
  (2002). <https://dl.acm.org/doi/10.1145/582415.582418>
  nDCG, used because its rank discount is what makes the metric sensitive to
  reranking.
- Voorhees, *The TREC-8 Question Answering Track Report* (1999).
  Mean reciprocal rank.
- Known-item retrieval as an evaluation design: using a document's own title as
  the query avoids needing human relevance judgements, which were not
  obtainable for 17,558 chunks at this scale. Discussed in
  [`../capstone/docs/design-decisions.md`](../capstone/docs/design-decisions.md#19-known-item-retrieval-evaluation).
