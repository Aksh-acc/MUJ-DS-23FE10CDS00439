"""Cross-encoder reranking of fused candidates.

A bi-encoder embeds the query and the passage independently, so the two never
see each other: the similarity is between two summaries written in ignorance.
That is what makes it fast enough to index 16k chunks, and also what makes it
blunt at the top of the ranking, where the surviving candidates are all roughly
on topic and the question is which one actually answers the query.

A cross-encoder scores the pair jointly, letting every query token attend to
every passage token. It is far more accurate at that fine distinction and far
too slow to run over a whole branch, so it is applied only to the fused top
``candidates`` per branch. Cost is therefore bounded by configuration rather
than by corpus size, which is what makes the two-stage design worth its
complexity.

``ms-marco-MiniLM-L-6-v2`` is trained on MS MARCO web search relevance, which is
a reasonable proxy for the ad-hoc question answering this system does, and it
keeps the same 6-layer MiniLM budget as the retrieval encoder.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from ..config import RerankerConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_cross_encoder(name: str) -> CrossEncoder:
    logger.info("loading cross-encoder %s", name)
    return CrossEncoder(name)


class Reranker:
    """Reorders candidates by cross-encoder relevance, with graceful degradation.

    If the model cannot be loaded the reranker reports itself unavailable and the
    retriever keeps the fused order. Losing reranking costs precision; failing
    the whole query costs the answer.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self._model: CrossEncoder | None = None
        self._failed = False

    @property
    def available(self) -> bool:
        if not self.config.enabled or self._failed:
            return False
        return self._ensure_model() is not None

    def _ensure_model(self) -> CrossEncoder | None:
        if self._model is None and not self._failed:
            try:
                self._model = _load_cross_encoder(self.config.model)
            except Exception as error:
                logger.warning(
                    "cross-encoder %s unavailable (%s); keeping fused order",
                    self.config.model,
                    error,
                )
                self._failed = True
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float] | None:
        """Return one relevance score per passage, or None if unavailable."""
        if not passages:
            return []
        model = self._ensure_model() if self.config.enabled else None
        if model is None:
            return None
        try:
            scores = model.predict(
                [(query, passage) for passage in passages],
                show_progress_bar=False,
            )
        except Exception as error:
            logger.warning("rerank failed (%s); keeping fused order", error)
            self._failed = True
            return None
        return [float(score) for score in scores]
