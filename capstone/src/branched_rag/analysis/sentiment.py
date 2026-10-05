"""Sentiment aggregation over retrieved customer reviews.

A language model asked "how do customers feel about this" will produce a
confident adjective, but nothing countable and nothing reproducible. This module
adds a measured number alongside it: a supervised classifier scores each
retrieved review chunk and the distribution is reported.

``distilbert-base-uncased-finetuned-sst-2-english`` is used because it is a
small, well-known binary sentiment model, and because binary is the honest
granularity here -- the reviews branch is drawn from a dataset whose own labels
are binary, so a five-point prediction would imply precision the source does not
have.

Two deliberate limits:

* Only the reviews branch is scored. SST-2 is trained on opinionated prose;
  applied to a Reuters earnings report it returns a confident label for text
  that carries no sentiment at all, which would be a measurement of nothing.
* The aggregate describes the *retrieved* reviews, not the market. It is
  reported next to the sample size so it cannot be read as a population
  estimate, and the corpus is polarity-balanced at ingest so the retrieved mix
  reflects the query rather than corpus construction.

The result feeds the fusion prompt as a quantitative note, so the final answer
can cite a proportion instead of an impression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from ..config import SentimentConfig
from ..retrieval.retriever import Evidence

logger = logging.getLogger(__name__)

# SST-2 truncates at 512 tokens; chunks are shorter than this but the guard keeps
# a long financial chunk from raising if the branch filter is ever relaxed.
_MAX_CHARS = 1200


@dataclass
class SentimentReport:
    positive: int
    negative: int
    sampled: int
    mean_confidence: float
    available: bool = True
    error: str = ""

    @property
    def positive_share(self) -> float:
        return self.positive / self.sampled if self.sampled else 0.0

    @property
    def verdict(self) -> str:
        if not self.sampled:
            return "no reviews retrieved"
        share = self.positive_share
        if share >= 0.7:
            return "mostly positive"
        if share >= 0.55:
            return "leaning positive"
        if share > 0.45:
            return "mixed"
        if share > 0.3:
            return "leaning negative"
        return "mostly negative"

    def as_note(self) -> str:
        """One line for the fusion prompt."""
        if not self.available or not self.sampled:
            return ""
        return (
            f"A sentiment classifier scored the {self.sampled} retrieved review "
            f"passages: {self.positive} positive, {self.negative} negative "
            f"({self.positive_share:.0%} positive, {self.verdict}). This describes "
            f"the retrieved sample only, not the whole market."
        )


@lru_cache(maxsize=2)
def _load_classifier(model: str):
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    logger.info("loading sentiment model %s", model)
    return pipeline("sentiment-analysis", model=model, device=device, truncation=True)


class SentimentAnalyzer:
    def __init__(self, config: SentimentConfig) -> None:
        self.config = config
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled and not self._failed

    def analyse(self, evidence: list[Evidence]) -> SentimentReport:
        """Score review-branch evidence and return the label distribution."""
        reviews = [item for item in evidence if item.branch == "reviews"]
        if not self.enabled or not reviews:
            return SentimentReport(
                positive=0,
                negative=0,
                sampled=0,
                mean_confidence=0.0,
                available=self.enabled,
                error="" if self.enabled else "sentiment analysis disabled",
            )

        texts = [item.text[:_MAX_CHARS] for item in reviews[: self.config.max_samples]]
        try:
            predictions = _load_classifier(self.config.model)(texts)
        except Exception as error:
            self._failed = True
            logger.warning("sentiment model unavailable: %s", error)
            return SentimentReport(
                positive=0,
                negative=0,
                sampled=0,
                mean_confidence=0.0,
                available=False,
                error=str(error),
            )

        positive = sum(1 for p in predictions if p["label"].upper().startswith("POS"))
        confidence = sum(float(p["score"]) for p in predictions) / len(predictions)
        return SentimentReport(
            positive=positive,
            negative=len(predictions) - positive,
            sampled=len(predictions),
            mean_confidence=confidence,
        )
