"""Sentence embedding with a single cached model instance.

Model choice: ``sentence-transformers/all-MiniLM-L6-v2``.

It is 22M parameters and 384 dimensions, roughly a tenth the size of
``all-mpnet-base-v2`` while retaining most of its retrieval quality on short
passages. That matters here for two concrete reasons rather than as a general
preference: the index is rebuilt from scratch during marking and demos, so
encoding ~12k chunks has to finish in minutes on whatever hardware is present;
and 384-dimension vectors keep the whole index small enough to hold in memory
alongside a reranker and a generator.

Vectors are L2-normalised at encode time so that a FAISS inner-product index
computes exact cosine similarity, which keeps scores comparable across branches
of very different document length.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EmbeddingConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_model(name: str) -> SentenceTransformer:
    logger.info("loading embedding model %s", name)
    return SentenceTransformer(name)


class Embedder:
    """Thin wrapper that fixes normalisation and batching policy in one place."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model = _load_model(config.model)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
