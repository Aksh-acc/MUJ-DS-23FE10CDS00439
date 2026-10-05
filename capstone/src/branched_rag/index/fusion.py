"""Reciprocal Rank Fusion for combining dense and sparse result lists.

Dense cosine similarity lands in roughly [0, 1] with most real hits bunched
between 0.3 and 0.7. BM25 is unbounded and its scale depends on corpus
statistics, so the same raw score means different things in the reviews branch
and the financials branch. Any weighted sum of the two raw scores is therefore
comparing incommensurable units, and min-max normalising per query makes the
weights depend on how good the best hit happened to be.

RRF sidesteps this by discarding the scores and keeping only the ranks:

    score(d) = sum over lists of  weight_list / (k + rank_list(d))

Rank is scale-free, so no normalisation is needed and the branch weights in
config.yaml mean the same thing everywhere. ``k`` damps the contribution of
deep positions; k=60 is the value from the original RRF paper and is used here
because it is the published default rather than something tuned on this corpus.

A document found by both retrievers accumulates both terms, so agreement between
the two systems is rewarded without ever having to compare their scores.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedHit:
    row: int
    score: float
    dense_rank: int | None
    sparse_rank: int | None

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.sparse_rank is not None


def reciprocal_rank_fusion(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Fuse two ranked ``(row, score)`` lists into one ranking.

    Ranks are 1-based. Input scores are used only for ordering, which the callers
    have already applied, and are otherwise ignored by design.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[int, float] = {}
    dense_ranks: dict[int, int] = {}
    sparse_ranks: dict[int, int] = {}

    for rank, (row, _) in enumerate(dense, start=1):
        dense_ranks[row] = rank
        scores[row] = scores.get(row, 0.0) + dense_weight / (k + rank)

    for rank, (row, _) in enumerate(sparse, start=1):
        sparse_ranks[row] = rank
        scores[row] = scores.get(row, 0.0) + sparse_weight / (k + rank)

    hits = [
        FusedHit(
            row=row,
            score=score,
            dense_rank=dense_ranks.get(row),
            sparse_rank=sparse_ranks.get(row),
        )
        for row, score in scores.items()
    ]
    # Ties broken by row index so the ordering is deterministic across runs.
    hits.sort(key=lambda hit: (-hit.score, hit.row))
    return hits[:top_k] if top_k is not None else hits
