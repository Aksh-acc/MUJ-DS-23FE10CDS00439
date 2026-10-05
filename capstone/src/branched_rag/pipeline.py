"""The public entry point: one call from question to cited answer.

``BranchedRAGPipeline.answer()`` runs the full sequence and returns every
intermediate stage alongside the final text. The app and the evaluation harness
both consume this one object, so what the UI displays is exactly what the
metrics are computed from -- there is no second, divergent code path for the
demo.

Stage order, with the timing of each recorded:

    route -> retrieve per branch -> sentiment -> summarise per branch -> fuse

Every stage degrades rather than raises. A missing reranker loses precision, a
missing generator leaves retrieval working as an evidence browser, and a failed
fusion call falls back to concatenated branch summaries. A capstone demo that
dies on a cold cache or an expired key is worth less than one that says what it
could not do.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .analysis.sentiment import SentimentAnalyzer, SentimentReport
from .config import Config, load_config
from .generation.llm import LLMProvider, build_llm
from .generation.synthesis import FusedAnswer, Synthesizer
from .index.embedder import Embedder
from .index.store import IndexStore
from .retrieval.query_ops import QueryExpander, extract_keyphrases
from .retrieval.reranker import Reranker
from .retrieval.retriever import BranchResult, BranchRetriever, Evidence
from .retrieval.router import BranchRouter, RoutingPlan

logger = logging.getLogger(__name__)


@dataclass
class PipelineAnswer:
    query: str
    plan: RoutingPlan
    results: list[BranchResult]
    answer: FusedAnswer
    sentiment: SentimentReport
    keyphrases: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def evidence(self) -> list[Evidence]:
        return [item for result in self.results for item in result.evidence]

    @property
    def evidence_by_citation(self) -> dict[int, Evidence]:
        return {item.citation: item for item in self.evidence}

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def cited_evidence(self) -> list[Evidence]:
        """Evidence the final answer actually used, in citation order."""
        used = set(self.answer.citations)
        return [item for item in self.evidence if item.citation in used]

    @property
    def grounding_rate(self) -> float:
        """Share of the answer's citations that resolve to supplied evidence.

        Citations are filtered against the retrieved set during synthesis, so in
        normal operation this is 1.0. It is surfaced because a drop below 1.0
        means the model invented a source number, which is the failure this
        design exists to make visible.
        """
        if not self.answer.citations:
            return 0.0
        valid = set(self.evidence_by_citation)
        return sum(1 for c in self.answer.citations if c in valid) / len(
            self.answer.citations
        )


class BranchedRAGPipeline:
    def __init__(
        self,
        config: Config,
        store: IndexStore,
        embedder: Embedder,
        llm: LLMProvider,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.llm = llm

        self.router = BranchRouter(config, embedder, llm=llm)
        self.expander = QueryExpander(embedder, llm=llm)
        self.retriever = BranchRetriever(
            config, store, self.expander, Reranker(config.reranker)
        )
        self.synthesizer = Synthesizer(config, llm)
        self.sentiment = SentimentAnalyzer(config.sentiment)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "BranchedRAGPipeline":
        """Build a pipeline from config.yaml and the persisted indexes."""
        config = load_config(config_path)
        if not IndexStore.exists(config):
            raise FileNotFoundError(
                f"no index in {config.paths.artifacts}. Run:\n"
                "  python scripts/build_corpus.py\n"
                "  python scripts/build_index.py"
            )
        embedder = Embedder(config.embedding)
        return cls(
            config=config,
            store=IndexStore.load(config),
            embedder=embedder,
            llm=build_llm(config),
        )

    @property
    def generation_available(self) -> bool:
        return self.llm.available

    def answer(
        self,
        query: str,
        strategy: str | None = None,
        branches: list[str] | None = None,
        expand_with_llm: bool = False,
        generate: bool = True,
    ) -> PipelineAnswer:
        """Answer a question.

        ``branches`` overrides the router, which is what the UI uses to let a
        user inspect a branch the router rejected. ``generate=False`` stops after
        retrieval, which is how the retrieval metrics are measured without
        spending tokens.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        timings: dict[str, float] = {}

        started = time.perf_counter()
        plan = self.router.route(query, strategy=strategy)
        if branches:
            plan = _override_branches(plan, branches)
        timings["route"] = time.perf_counter() - started

        started = time.perf_counter()
        results = self.retriever.retrieve(plan, expand_with_llm=expand_with_llm)
        timings["retrieve"] = time.perf_counter() - started

        started = time.perf_counter()
        keyphrases = extract_keyphrases(query, self.embedder, top_n=5)
        timings["keyphrases"] = time.perf_counter() - started

        started = time.perf_counter()
        sentiment = self.sentiment.analyse(
            [item for result in results for item in result.evidence]
        )
        timings["sentiment"] = time.perf_counter() - started

        if not generate:
            return PipelineAnswer(
                query=query,
                plan=plan,
                results=results,
                answer=FusedAnswer(answer="", branch_summaries=[]),
                sentiment=sentiment,
                keyphrases=keyphrases,
                timings=timings,
            )

        started = time.perf_counter()
        summaries = self.synthesizer.summarise_branches(query, results)
        timings["branch_synthesis"] = time.perf_counter() - started

        started = time.perf_counter()
        fused = self.synthesizer.fuse(
            query, summaries, sentiment_note=sentiment.as_note() or None
        )
        timings["fusion"] = time.perf_counter() - started

        return PipelineAnswer(
            query=query,
            plan=plan,
            results=results,
            answer=fused,
            sentiment=sentiment,
            keyphrases=keyphrases,
            timings=timings,
        )


def _override_branches(plan: RoutingPlan, branches: list[str]) -> RoutingPlan:
    """Return a plan with ``branches`` selected, preserving the router's scores."""
    wanted = set(branches)
    return RoutingPlan(
        query=plan.query,
        strategy=f"{plan.strategy} (manual override)",
        decisions=[
            replace(decision, selected=decision.branch in wanted)
            for decision in plan.decisions
        ],
        fallback_applied=plan.fallback_applied,
        cutoff=plan.cutoff,
    )
