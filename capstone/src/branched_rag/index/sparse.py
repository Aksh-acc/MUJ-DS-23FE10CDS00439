"""Lexical retrieval with BM25 Okapi.

A dense bi-encoder maps "EUR 47.5 million" and "EUR 52.1 million" to almost the
same point: the embedding captures that both are European currency amounts and
discards which one. Market intelligence questions turn on exactly those tokens,
along with tickers, model numbers and company names that are rare enough to be
poorly represented in a 22M-parameter encoder. BM25 scores a term by how rare it
is in the collection, so it is strongest precisely where the dense index is
weakest. The two are combined in ``fusion.py`` rather than used in isolation.

The index is rebuilt from ``chunks.parquet`` at load time instead of being
pickled. Tokenising ~12k short chunks takes under a second, and a pickled BM25
object silently breaks whenever ``rank_bm25`` or the tokeniser changes.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'.$%-]*")

# A deliberately short stop list. Aggressive stopping hurts here: "no", "not" and
# "down" carry the signal in a complaint or a profit warning, and BM25's own IDF
# term already discounts anything that appears everywhere.
_STOPWORDS = frozenset(
    "a an the and or but if then than that this these those of in on at to for "
    "from by with as is are was were be been being it its i you he she they we "
    "my your their our there here what which who whom when where how why all any "
    "both each more most other some such only own same so too very can will just "
    "do does did doing have has had having would could should may might must "
    "about into over under again further once".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, keep alphanumeric tokens, drop stopwords and bare punctuation.

    Internal ``'``, ``.``, ``$``, ``%`` and ``-`` are kept inside tokens so that
    ``q3``, ``47.5``, ``$55.8bn``, ``8%`` and ``isn't`` survive as single units.
    """
    tokens = _TOKEN.findall(text.lower())
    return [
        token.strip(".-'")
        for token in tokens
        if token not in _STOPWORDS and len(token.strip(".-'")) > 1
    ]


class SparseIndex:
    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        self._size = len(corpus_tokens)
        # BM25Okapi cannot be constructed on an empty corpus.
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def build(cls, texts: list[str]) -> "SparseIndex":
        return cls([tokenize(text) for text in texts])

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return ``(row, score)`` pairs ordered by descending BM25 score."""
        if self._bm25 is None or top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(row, float(score)) for row, score in ranked[:top_k] if score > 0.0]
