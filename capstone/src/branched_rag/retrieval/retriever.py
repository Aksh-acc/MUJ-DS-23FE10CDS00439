"""Per-branch retrieval: hybrid search, rank fusion and reranking.

One branch is retrieved by this sequence:

1. expand the query with that branch's own vocabulary (``query_ops``)
2. search the branch's dense index and its BM25 index independently
3. fuse the two rankings with RRF, weighted by the branch's dense/sparse split
4. rerank the fused top candidates with a cross-encoder
5. keep the branch's own ``top_k``

Branches are retrieved concurrently. The work is dominated by two model forward
passes and a BM25 scan, all of which release the GIL inside native code, so a
thread pool gives real wall-clock savings without the overhead and pickling
constraints of separate processes.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import Config
from ..index.fusion import reciprocal_rank_fusion
from ..index.store import BranchIndex, IndexStore
from .query_ops import BranchQuery, QueryExpander
from .reranker import Reranker
from .router import RoutingPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk, with the provenance needed to cite and audit it."""

    citation: int
    chunk_id: str
    doc_id: str
    branch: str
    title: str
    text: str
    url: str
    source: str
    fused_score: float
    rerank_score: float | None
    dense_rank: int | None
    sparse_rank: int | None
    extra: dict = field(default_factory=dict)

    @property
    def found_by_both(self) -> bool:
        return self.dense_rank is not None and self.sparse_rank is not None

    @property
    def retrieval_path(self) -> str:
        if self.found_by_both:
            return "dense + lexical"
        if self.dense_rank is not None:
            return "dense only"
        if self.sparse_rank is not None:
            return "lexical only"
        return "unknown"


@dataclass
class BranchResult:
    branch: str
    label: str
    query: BranchQuery
    evidence: list[Evidence]
    reranked: bool
    candidates_considered: int

    @property
    def is_empty(self) -> bool:
        return not self.evidence


class BranchRetriever:
    def __init__(
        self,
        config: Config,
        store: IndexStore,
        expander: QueryExpander,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.expander = expander
        self.reranker = reranker or Reranker(config.reranker)

    def retrieve(
        self,
        plan: RoutingPlan,
        expand_with_llm: bool = False,
        citation_start: int = 1,
    ) -> list[BranchResult]:
        """Retrieve every branch the plan selected, concurrently."""
        branches = [name for name in plan.selected if name in self.store]
        if not branches:
            logger.warning("routing plan selected no indexed branch")
            return []

        with ThreadPoolExecutor(max_workers=len(branches)) as pool:
            results = list(
                pool.map(
                    lambda name: self._retrieve_branch(
                        name, plan.query, expand_with_llm
                    ),
                    branches,
                )
            )

        # Citations are numbered after retrieval so the numbers run in branch
        # order and stay stable regardless of which thread finished first.
        counter = citation_start
        for result in results:
            renumbered = []
            for item in result.evidence:
                renumbered.append(
                    Evidence(**{**item.__dict__, "citation": counter})
                )
                counter += 1
            result.evidence = renumbered
        return results

    def _retrieve_branch(
        self, name: str, query: str, expand_with_llm: bool
    ) -> BranchResult:
        index: BranchIndex = self.store[name]
        branch = index.config
        branch_query = self.expander.expand(query, branch, use_llm=expand_with_llm)

        retrieval = self.config.retrieval
        dense_hits = index.dense.search(
            self.expander.embedder.encode_one(branch_query.text),
            retrieval.dense_candidates,
        )
        sparse_hits = index.sparse.search(
            branch_query.text, retrieval.sparse_candidates
        )

        fused = reciprocal_rank_fusion(
            dense=dense_hits,
            sparse=sparse_hits,
            dense_weight=branch.dense_weight,
            sparse_weight=branch.sparse_weight,
            k=retrieval.rrf_k,
            top_k=max(self.config.reranker.candidates, branch.top_k),
        )
        if not fused:
            return BranchResult(
                branch=name,
                label=branch.label,
                query=branch_query,
                evidence=[],
                reranked=False,
                candidates_considered=0,
            )

        records = [index.record(hit.row) for hit in fused]
        rerank_scores = self.reranker.score(
            branch_query.text, [record["text"] for record in records]
        )

        order = range(len(fused))
        if rerank_scores is not None:
            order = sorted(order, key=lambda i: -rerank_scores[i])

        evidence = [
            Evidence(
                citation=0,  # assigned by retrieve()
                chunk_id=records[i]["chunk_id"],
                doc_id=records[i]["doc_id"],
                branch=name,
                title=records[i]["title"],
                text=records[i]["text"],
                url=records[i]["url"],
                source=records[i]["source"],
                fused_score=fused[i].score,
                rerank_score=rerank_scores[i] if rerank_scores is not None else None,
                dense_rank=fused[i].dense_rank,
                sparse_rank=fused[i].sparse_rank,
                extra=records[i]["extra"],
            )
            for i in list(order)[: branch.top_k]
        ]

        return BranchResult(
            branch=name,
            label=branch.label,
            query=branch_query,
            evidence=evidence,
            reranked=rerank_scores is not None,
            candidates_considered=len(fused),
        )
