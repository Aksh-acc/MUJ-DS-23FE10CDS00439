"""Query-side NLP: normalisation, keyphrase extraction and branch-conditioned expansion.

Branch-conditioned expansion is the reason this module exists. The same question
has to be asked differently of each corpus, because the three corpora use
different vocabulary for the same concept. "Are buyers happy with battery life"
is phrased as "died after two months" in a review and as "handset margin
pressure" in a financial report. Sending the identical query string to all three
indexes wastes the structure that branching created.

The default expansion is deterministic: for each selected branch, append the
branch-vocabulary terms that are themselves closest to the query in embedding
space. The expansion vocabulary is the branch's own keyword list from
config.yaml, so a term can only be added if the author of the branch declared it
relevant. That keeps expansion auditable -- the UI can show exactly which terms
were added and why -- and avoids an LLM round trip per branch before retrieval
has started. An LLM rewrite is available for comparison and falls back to the
deterministic path on any error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ..config import BranchConfig
from ..index.embedder import Embedder

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_EXPANSION_TERMS = 3


@dataclass(frozen=True)
class BranchQuery:
    branch: str
    text: str
    added_terms: tuple[str, ...]
    method: str

    @property
    def was_expanded(self) -> bool:
        return bool(self.added_terms)


def normalize_query(query: str) -> str:
    return _WHITESPACE.sub(" ", query or "").strip()


def _is_near_duplicate(term: str, accepted: list[str]) -> bool:
    """True if ``term`` is a morphological variant of something already accepted.

    A prefix test rather than a stemmer: the comparison is only ever between short
    config-declared keywords, where "cost"/"costs" and "margin"/"margins" are the
    entire problem, and adding a stemmer dependency to solve it would be absurd.
    """
    return any(
        term.startswith(other) or other.startswith(term) for other in accepted
    )


def extract_keyphrases(
    text: str,
    embedder: Embedder,
    top_n: int = 6,
    diversity: float = 0.6,
    ngram_range: tuple[int, int] = (1, 3),
) -> list[str]:
    """Rank candidate n-grams by similarity to the whole text, then diversify.

    This is the KeyBERT formulation: embed the text, embed each candidate phrase,
    and select by cosine similarity. Maximal Marginal Relevance then trades some
    relevance for coverage, because the top phrases by raw similarity are usually
    near-duplicates of each other ("battery life", "the battery life", "battery").

    It reuses the retrieval embedder rather than adding a separate keyphrase
    model, so this costs one extra forward pass over a few dozen short strings.
    """
    text = normalize_query(text)
    if not text:
        return []

    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
        candidates = vectorizer.fit(['%s' % text]).get_feature_names_out().tolist()
    except ValueError:
        # Raised when the text is entirely stopwords, which is a legitimate query.
        return []
    if not candidates:
        return []

    text_vector = embedder.encode_one(text)
    candidate_vectors = embedder.encode(candidates)
    relevance = candidate_vectors @ text_vector

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < min(top_n, len(candidates)):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            chosen_vectors = candidate_vectors[selected]
            best = max(
                remaining,
                key=lambda i: (1 - diversity) * relevance[i]
                - diversity * float(np.max(candidate_vectors[i] @ chosen_vectors.T)),
            )
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


class QueryExpander:
    def __init__(self, embedder: Embedder, llm=None) -> None:
        self.embedder = embedder
        self.llm = llm
        self._vocab_cache: dict[str, np.ndarray] = {}

    def _vocab_vectors(self, branch: BranchConfig) -> np.ndarray:
        if branch.name not in self._vocab_cache:
            self._vocab_cache[branch.name] = self.embedder.encode(
                list(branch.keywords)
            )
        return self._vocab_cache[branch.name]

    def expand(
        self, query: str, branch: BranchConfig, use_llm: bool = False
    ) -> BranchQuery:
        query = normalize_query(query)
        if use_llm and self.llm is not None and self.llm.available:
            rewritten = self._expand_with_llm(query, branch)
            if rewritten is not None:
                return rewritten

        query_vector = self.embedder.encode_one(query)
        similarity = self._vocab_vectors(branch) @ query_vector
        query_terms = set(query.lower().split())

        ranked = sorted(
            range(len(branch.keywords)),
            key=lambda i: -float(similarity[i]),
        )

        # Keyword lists carry both singular and plural forms, and they embed almost
        # identically, so the top-n by similarity was spending slots on "cost" and
        # "costs". Near-duplicates are skipped by prefix so the expansion adds
        # three distinct concepts rather than three spellings of one.
        added: list[str] = []
        for i in ranked:
            term = branch.keywords[i]
            if term in query_terms or _is_near_duplicate(term, added):
                continue
            added.append(term)
            if len(added) == _EXPANSION_TERMS:
                break

        text = f"{query} {' '.join(added)}".strip() if added else query
        return BranchQuery(
            branch=branch.name,
            text=text,
            added_terms=tuple(added),
            method="vocabulary",
        )

    def _expand_with_llm(self, query: str, branch: BranchConfig) -> BranchQuery | None:
        prompt = (
            f"Rewrite the question as a search query for this source.\n\n"
            f"Source: {branch.label} -- {branch.description.strip()}\n"
            f"Question: {query}\n\n"
            "Reply with the search query only, on one line, no explanation."
        )
        try:
            reply = self.llm.complete(
                prompt,
                system="You rewrite questions into search queries. Reply with one line.",
                temperature=0.0,
                max_tokens=80,
            )
        except Exception as error:
            logger.warning("llm expansion failed for branch %s: %s", branch.name, error)
            return None

        text = normalize_query(reply.splitlines()[0] if reply else "")
        if not text or len(text) > 400:
            return None
        return BranchQuery(
            branch=branch.name, text=text, added_terms=(), method="llm"
        )
