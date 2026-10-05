# Notebooks

Analysis notebooks for the capstone project. All three are committed with their
outputs intact, so the results can be read without re-running anything.

| Notebook | What it establishes |
|---|---|
| `01_corpus_exploration.ipynb` | The branch imbalance that motivates the architecture: equal documents in, 5.48 chunks per financial article against 1.02 per news item. Also verifies 0 of 17,558 chunks retain a raw-dataset artifact, and that the reviews branch is polarity-balanced. |
| `02_retrieval_experiments.ipynb` | That dense and lexical retrieval disagree on the same query, why their raw scores cannot be summed, and the full dense / sparse / fused / fused+rerank ablation. |
| `03_routing_and_evaluation.ipynb` | Clause splitting on a compound question, the three router strategies compared on 35 labelled queries, the routing errors, and the controlled flat-versus-branched comparison. |

## Running them

They import the project package from `../capstone/src`, so the capstone index
must be built first:

```bash
cd ../capstone
pip install -r requirements.txt
python scripts/build_corpus.py
python scripts/build_index.py
```

Then open any notebook from this directory. Only notebook 02 and 03 load models;
notebook 01 reads the committed parquet files and needs nothing else.
