"""Evaluation harness: routing accuracy, retrieval quality, and the flat baseline.

Three experiments, each answering a question the design claims an answer to.

**Routing accuracy** against the hand-labelled set in ``data/eval/queries.jsonl``.
Labels record which sources a question *should* consult, judged from the
question's wording and independently of what the router currently does, so the
numbers can disagree with the implementation. The three strategies are scored
side by side to show whether combining the embedding and lexical signals
actually beats either alone.

**Known-item retrieval**, which needs no human relevance judgements. A document
title is used as the query and the document's own chunks are the relevant set:
if asking for a document by its headline does not return it, retrieval is
broken. Generic titles ("Market report") are excluded because they identify
nothing. Running this over dense-only, sparse-only and fused retrieval turns the
hybrid design from an assertion into a measurement.

**Flat versus branched**, the controlled comparison the whole project rests on.
The baseline reuses the same embedding vectors, the same BM25 implementation,
the same RRF and the same total evidence budget; the only difference is that its
index is one pool instead of three. Vectors are read back out of the existing
FAISS indexes rather than recomputed, so the two systems are guaranteed to be
compared on identical representations.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..index.dense import DenseIndex
from ..index.fusion import reciprocal_rank_fusion
from ..index.sparse import SparseIndex
from ..index.store import IndexStore
from ..pipeline import BranchedRAGPipeline
from .metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank, set_score, source_coverage

logger = logging.getLogger(__name__)

# Titles the source datasets use as placeholders; useless as known-item queries.
_GENERIC_TITLES = frozenset(
    {"market report", "customer review", "financial report", ""}
)


@dataclass
class EvalQuery:
    id: str
    query: str
    branches: set[str]
    kind: str


def load_eval_queries(path: Path) -> list[EvalQuery]:
    if not path.exists():
        raise FileNotFoundError(f"eval query set not found: {path}")
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries.append(
                EvalQuery(
                    id=row["id"],
                    query=row["query"],
                    branches=set(row["branches"]),
                    kind=row.get("kind", "single"),
                )
            )
    return queries


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def evaluate_router(
    pipeline: BranchedRAGPipeline,
    queries: list[EvalQuery],
    strategies: tuple[str, ...] = ("embedding", "lexical", "hybrid"),
) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        for subset in ("all", "single", "multi"):
            selected = [
                query for query in queries if subset == "all" or query.kind == subset
            ]
            if not selected:
                continue
            scores = [
                set_score(
                    set(pipeline.router.route(query.query, strategy=strategy).selected),
                    query.branches,
                )
                for query in selected
            ]
            rows.append(
                {
                    "strategy": strategy,
                    "subset": subset,
                    "queries": len(selected),
                    "exact_match": round(
                        mean([float(s.exact_match) for s in scores]), 3
                    ),
                    "precision": round(mean([s.precision for s in scores]), 3),
                    "recall": round(mean([s.recall for s in scores]), 3),
                    "f1": round(mean([s.f1 for s in scores]), 3),
                }
            )
    return pd.DataFrame(rows)


def router_errors(
    pipeline: BranchedRAGPipeline, queries: list[EvalQuery], strategy: str = "hybrid"
) -> pd.DataFrame:
    rows = []
    for query in queries:
        predicted = set(pipeline.router.route(query.query, strategy=strategy).selected)
        if predicted != query.branches:
            rows.append(
                {
                    "id": query.id,
                    "query": query.query[:62],
                    "expected": ",".join(sorted(query.branches)),
                    "predicted": ",".join(sorted(predicted)) or "(none)",
                    "missed": ",".join(sorted(query.branches - predicted)) or "-",
                    "spurious": ",".join(sorted(predicted - query.branches)) or "-",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Known-item retrieval
# --------------------------------------------------------------------------


def _sample_known_items(
    store: IndexStore, branch: str, sample_size: int, seed: int
) -> list[tuple[str, set[str]]]:
    chunks = store[branch].chunks
    usable = chunks[~chunks["title"].str.lower().str.strip().isin(_GENERIC_TITLES)]
    # One query per document, not per chunk, or long documents dominate the sample.
    by_doc = usable.groupby("doc_id")
    doc_ids = sorted(by_doc.groups)
    if not doc_ids:
        return []
    rng = random.Random(seed)
    chosen = rng.sample(doc_ids, min(sample_size, len(doc_ids)))
    items = []
    for doc_id in chosen:
        group = by_doc.get_group(doc_id)
        title = str(group.iloc[0]["title"]).strip()
        if len(title) < 12:
            continue
        items.append((title, set(group["chunk_id"])))
    return items


def evaluate_retrieval(
    pipeline: BranchedRAGPipeline,
    sample_size: int = 120,
    k: int = 10,
    seed: int = 20251006,
) -> pd.DataFrame:
    """Score dense, sparse, fused and fused+rerank retrieval per branch."""
    store = pipeline.store
    embedder = pipeline.embedder
    retrieval = pipeline.config.retrieval
    rows = []

    for branch in store.names:
        index = store[branch]
        chunk_ids = index.chunks["chunk_id"].tolist()
        items = _sample_known_items(store, branch, sample_size, seed)
        if not items:
            continue

        modes: dict[str, list[list[str]]] = {
            "dense": [],
            "sparse": [],
            "fused": [],
            "fused+rerank": [],
        }
        relevant_sets = []

        for title, relevant in items:
            relevant_sets.append(relevant)
            query_vector = embedder.encode_one(title)

            dense_hits = index.dense.search(query_vector, retrieval.dense_candidates)
            sparse_hits = index.sparse.search(title, retrieval.sparse_candidates)

            modes["dense"].append([chunk_ids[row] for row, _ in dense_hits[:k]])
            modes["sparse"].append([chunk_ids[row] for row, _ in sparse_hits[:k]])

            fused = reciprocal_rank_fusion(
                dense=dense_hits,
                sparse=sparse_hits,
                dense_weight=index.config.dense_weight,
                sparse_weight=index.config.sparse_weight,
                k=retrieval.rrf_k,
                top_k=max(pipeline.config.reranker.candidates, k),
            )
            modes["fused"].append([chunk_ids[hit.row] for hit in fused[:k]])

            scores = pipeline.retriever.reranker.score(
                title, [index.chunks.iloc[hit.row]["text"] for hit in fused]
            )
            if scores is None:
                modes["fused+rerank"].append([chunk_ids[hit.row] for hit in fused[:k]])
            else:
                order = sorted(range(len(fused)), key=lambda i: -scores[i])
                modes["fused+rerank"].append(
                    [chunk_ids[fused[i].row] for i in order[:k]]
                )

        for mode, runs in modes.items():
            rows.append(
                {
                    "branch": branch,
                    "mode": mode,
                    "queries": len(runs),
                    f"recall@{k}": round(
                        mean(
                            [
                                recall_at_k(run, relevant, k)
                                for run, relevant in zip(runs, relevant_sets)
                            ]
                        ),
                        3,
                    ),
                    "mrr": round(
                        mean(
                            [
                                reciprocal_rank(run, relevant)
                                for run, relevant in zip(runs, relevant_sets)
                            ]
                        ),
                        3,
                    ),
                    f"ndcg@{k}": round(
                        mean(
                            [
                                ndcg_at_k(run, relevant, k)
                                for run, relevant in zip(runs, relevant_sets)
                            ]
                        ),
                        3,
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Flat baseline
# --------------------------------------------------------------------------


@dataclass
class FlatBaseline:
    """A single pooled index over every chunk, for comparison against branching.

    Built from vectors read back out of the per-branch FAISS indexes, so both
    systems are compared on byte-identical embeddings.
    """

    chunks: pd.DataFrame
    dense: DenseIndex
    sparse: SparseIndex
    config: Config

    @classmethod
    def from_store(cls, store: IndexStore, config: Config) -> "FlatBaseline":
        frames, matrices = [], []
        for name in store.names:
            index = store[name]
            frames.append(index.chunks)
            matrices.append(index.dense.vectors())
        chunks = pd.concat(frames, ignore_index=True)
        vectors = np.ascontiguousarray(np.vstack(matrices), dtype="float32")
        logger.info("flat baseline: %d chunks pooled", len(chunks))
        return cls(
            chunks=chunks,
            dense=DenseIndex.build(vectors),
            sparse=SparseIndex.build(chunks["text"].tolist()),
            config=config,
        )

    def retrieve(self, query: str, embedder, top_n: int) -> list[str]:
        """Return the branches of the top ``top_n`` pooled results, in order."""
        retrieval = self.config.retrieval
        dense_hits = self.dense.search(
            embedder.encode_one(query), retrieval.dense_candidates
        )
        sparse_hits = self.sparse.search(query, retrieval.sparse_candidates)
        # Averaged weights: a pooled index has no per-branch profile to apply,
        # which is itself one of the things branching buys.
        fused = reciprocal_rank_fusion(
            dense=dense_hits,
            sparse=sparse_hits,
            dense_weight=0.6,
            sparse_weight=0.4,
            k=retrieval.rrf_k,
            top_k=top_n,
        )
        return [self.chunks.iloc[hit.row]["branch"] for hit in fused]


def compare_flat_and_branched(
    pipeline: BranchedRAGPipeline,
    queries: list[EvalQuery],
    baseline: FlatBaseline | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare source coverage of the branched pipeline against a flat index.

    Each system gets the same number of evidence slots per query: whatever the
    branched pipeline retrieved. Returns per-query detail and a summary.
    """
    baseline = baseline or FlatBaseline.from_store(pipeline.store, pipeline.config)
    rows = []

    for query in queries:
        answer = pipeline.answer(query.query, generate=False)
        branched = [item.branch for item in answer.evidence]
        budget = len(branched)
        if budget == 0:
            continue
        flat = baseline.retrieve(query.query, pipeline.embedder, budget)

        rows.append(
            {
                "id": query.id,
                "kind": query.kind,
                "expected_sources": len(query.branches),
                "evidence_budget": budget,
                "branched_coverage": round(source_coverage(branched, query.branches), 3),
                "flat_coverage": round(source_coverage(flat, query.branches), 3),
                "branched_sources": len(set(branched)),
                "flat_sources": len(set(flat)),
                "flat_top_branch_share": round(
                    max(flat.count(b) for b in set(flat)) / budget, 3
                ),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail

    summary = (
        detail.groupby("kind", sort=False)
        .agg(
            queries=("id", "count"),
            branched_coverage=("branched_coverage", "mean"),
            flat_coverage=("flat_coverage", "mean"),
            branched_sources=("branched_sources", "mean"),
            flat_sources=("flat_sources", "mean"),
            flat_top_branch_share=("flat_top_branch_share", "mean"),
        )
        .round(3)
        .reset_index()
    )
    return detail, summary
