"""Branch selection: deciding which corpora a question should be answered from.

This is the component that makes the system branched rather than merely
hybrid. Three strategies are implemented and selectable from config.yaml so
they can be compared on the labelled query set in ``data/eval/queries.jsonl``.

``embedding``
    Cosine similarity between the query vector and a vector per branch, built
    by embedding that branch's descriptor text from config.yaml. Needs no
    training data and no labelled routing examples, which matters because none
    exist for this corpus. It generalises to paraphrase: "how do buyers feel
    about build quality" routes to reviews without sharing a word with the
    keyword list.

``lexical``
    Fraction of the branch's keyword list present in the query. Blind to
    paraphrase, but it is the only strategy that reliably fires on a bare
    domain term such as "EPS" or "margin" that a 384-dimension encoder places
    near nothing in particular.

``hybrid`` (default)
    A weighted sum of the two. They fail in opposite directions, so the
    combination is more stable than either, and it stays deterministic and
    inspectable -- the UI shows the per-branch score and which signal produced
    it. That auditability is the reason the default is not the LLM router.

``llm``
    One call asking an open-weight model to pick branches, with a strict
    fallback to hybrid when the reply does not parse. Available for comparison;
    it costs a network round trip before retrieval has even started, and it
    cannot be evaluated offline.

Two refinements were added after measuring the first version on real queries.

*Clause-level scoring.* "Should we launch a budget tablet? Consider demand,
rivals and financial risk" is three questions wearing one coat. Encoding it as a
single vector averages three intents into a point near none of them, and the
first version selected only the news branch and silently dropped the financial
half of the question. The query is therefore split into clauses, every clause is
scored against every branch, and each branch keeps its best clause score. One
clause that clearly needs a branch is enough to select it, which is the correct
semantics for a compound question.

*Relative selection.* A fixed cutoff cannot work, because descriptor cosine
similarity for this encoder spans roughly 0.0-0.45 and the usable range shifts
with query length. A branch is kept when it scores within ``relative_threshold``
of the best branch, with ``min_score`` acting only as an absolute floor for
branches that are simply irrelevant. Selection is thus about which branches are
competitive for this query rather than about hitting a magic number.

A query that clears no threshold at all still has to be answered, so
``always_keep_top`` retains the single best branch rather than returning an
empty plan and no evidence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..index.embedder import Embedder
from ..index.sparse import tokenize

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

# Clause boundaries for compound questions: sentence punctuation, commas, and the
# coordinators that join independent asks. Splitting on these is crude compared
# with a dependency parse, but it needs no parser model and the failure mode is
# benign -- an over-split clause simply contributes a weaker score to the max.
_CLAUSE_SPLIT = re.compile(
    r"[;,.?!]+|\s+(?:and|or|plus|versus|vs|along with|as well as|also)\s+",
    re.IGNORECASE,
)


def split_clauses(query: str) -> list[str]:
    """Split a compound query into clauses, keeping any that carry a content word."""
    pieces = (piece.strip() for piece in _CLAUSE_SPLIT.split(query))
    return [piece for piece in pieces if piece and _WORD.search(piece.lower())]


@dataclass(frozen=True)
class BranchDecision:
    branch: str
    label: str
    score: float
    embedding_score: float
    lexical_score: float
    selected: bool
    matched_keywords: tuple[str, ...] = ()
    best_clause: str = ""

    @property
    def rationale(self) -> str:
        parts = [f"descriptor similarity {self.embedding_score:.2f}"]
        if self.matched_keywords:
            parts.append("terms: " + ", ".join(self.matched_keywords[:4]))
        if self.best_clause:
            parts.append(f'driven by "{self.best_clause}"')
        return "; ".join(parts)


@dataclass
class RoutingPlan:
    query: str
    strategy: str
    decisions: list[BranchDecision] = field(default_factory=list)
    fallback_applied: bool = False
    cutoff: float = 0.0

    @property
    def selected(self) -> list[str]:
        return [d.branch for d in self.decisions if d.selected]

    @property
    def selected_decisions(self) -> list[BranchDecision]:
        return [d for d in self.decisions if d.selected]

    def score_of(self, branch: str) -> float:
        for decision in self.decisions:
            if decision.branch == branch:
                return decision.score
        return 0.0


class BranchRouter:
    def __init__(self, config: Config, embedder: Embedder, llm=None) -> None:
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self._names = config.branch_names
        self._descriptors = embedder.encode(
            [config.branch(name).descriptor for name in self._names]
        )
        self._keywords = {
            name: frozenset(config.branch(name).keywords) for name in self._names
        }

    def route(self, query: str, strategy: str | None = None) -> RoutingPlan:
        strategy = strategy or self.config.router.strategy
        if strategy == "llm":
            plan = self._route_with_llm(query)
            if plan is not None:
                return plan
            logger.info("llm router unavailable or unparseable; using hybrid")
            strategy = "hybrid"
        return self._route_with_scores(query, strategy)

    def _embedding_scores(self, query: str) -> tuple[np.ndarray, list[str]]:
        """Score every branch against the query and each of its clauses.

        Returns the per-branch maximum and the clause that produced it, so the UI
        can show which part of a compound question pulled in each branch.
        """
        variants = [query]
        clauses = split_clauses(query)
        # A single-clause split is just the query again; skip the extra encoding.
        if len(clauses) > 1:
            variants.extend(clauses)

        # Both sides are unit vectors, so this matrix product is cosine similarity.
        similarity = self._descriptors @ self.embedder.encode(variants).T
        best_variant = similarity.argmax(axis=1)
        return (
            similarity.max(axis=1),
            [variants[position] for position in best_variant],
        )

    def _lexical_scores(self, query: str) -> tuple[np.ndarray, dict[str, list[str]]]:
        query_terms = set(_WORD.findall(query.lower())) | set(tokenize(query))
        scores = np.zeros(len(self._names), dtype="float32")
        matched: dict[str, list[str]] = {}
        for position, name in enumerate(self._names):
            keywords = self._keywords[name]
            hits = sorted(keywords & query_terms)
            matched[name] = hits
            # Normalised by a small constant, not by len(keywords): the lists are
            # different lengths, and dividing by them would make a branch with a
            # long keyword list structurally harder to select.
            scores[position] = min(len(hits) / 3.0, 1.0)
        return scores, matched

    def _route_with_scores(self, query: str, strategy: str) -> RoutingPlan:
        router = self.config.router
        embedding, best_clauses = self._embedding_scores(query)
        lexical, matched = self._lexical_scores(query)

        if strategy == "embedding":
            combined = embedding
        elif strategy == "lexical":
            combined = lexical
        elif strategy == "hybrid":
            combined = (
                router.embedding_weight * embedding + router.lexical_weight * lexical
            )
        else:
            raise ValueError(
                f"unknown router strategy {strategy!r}; "
                "expected embedding, lexical, hybrid or llm"
            )

        order = np.argsort(-combined)
        top_score = float(combined[order[0]])
        # Keep branches that are competitive with the best one. min_score is only
        # an absolute floor; see the module docstring on why a fixed cutoff fails.
        cutoff = max(router.min_score, top_score * router.relative_threshold)
        keep = {
            int(position)
            for position in order[: router.max_branches]
            if combined[position] >= cutoff
        }

        fallback_applied = False
        if not keep and router.always_keep_top:
            keep = {int(order[0])}
            fallback_applied = True

        decisions = [
            BranchDecision(
                branch=name,
                label=self.config.branch(name).label,
                score=float(combined[position]),
                embedding_score=float(embedding[position]),
                lexical_score=float(lexical[position]),
                selected=position in keep,
                matched_keywords=tuple(matched[name]),
                best_clause=best_clauses[position],
            )
            for position, name in enumerate(self._names)
        ]
        decisions.sort(key=lambda decision: -decision.score)

        return RoutingPlan(
            query=query,
            strategy=strategy,
            decisions=decisions,
            fallback_applied=fallback_applied,
            cutoff=cutoff,
        )

    def _route_with_llm(self, query: str) -> RoutingPlan | None:
        if self.llm is None or not self.llm.available:
            return None

        catalogue = "\n".join(
            f"- {name}: {self.config.branch(name).description.strip()}"
            for name in self._names
        )
        prompt = (
            "Select every source that is needed to answer the question.\n\n"
            f"Sources:\n{catalogue}\n\n"
            f"Question: {query}\n\n"
            'Reply with JSON only: {"branches": ["name", ...]}'
        )
        try:
            reply = self.llm.complete(
                prompt,
                system="You route questions to data sources. Reply with JSON only.",
                temperature=0.0,
                max_tokens=120,
            )
        except Exception as error:
            logger.warning("llm router call failed: %s", error)
            return None

        chosen = _parse_branch_list(reply, self._names)
        if not chosen:
            return None

        embedding, best_clauses = self._embedding_scores(query)
        lexical, matched = self._lexical_scores(query)
        decisions = [
            BranchDecision(
                branch=name,
                label=self.config.branch(name).label,
                score=1.0 if name in chosen else 0.0,
                embedding_score=float(embedding[position]),
                lexical_score=float(lexical[position]),
                selected=name in chosen,
                matched_keywords=tuple(matched[name]),
                best_clause=best_clauses[position],
            )
            for position, name in enumerate(self._names)
        ]
        decisions.sort(key=lambda decision: -decision.score)
        return RoutingPlan(query=query, strategy="llm", decisions=decisions)


def _parse_branch_list(reply: str, valid: list[str]) -> list[str]:
    """Pull a branch list out of an LLM reply, tolerating prose around the JSON."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            names = payload.get("branches", [])
            if isinstance(names, list):
                chosen = [n for n in names if n in valid]
                if chosen:
                    return chosen
        except json.JSONDecodeError:
            pass
    # Last resort: the model named the branches without valid JSON around them.
    lowered = reply.lower()
    return [name for name in valid if name in lowered]
