"""Shared fixtures.

The unit tests deliberately avoid loading real models. A fake embedder with the
same interface keeps the suite fast enough to run on every commit, and it lets
the router tests assert on exact scores, which is impossible against a real
encoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from branched_rag.config import load_config  # noqa: E402


class FakeEmbedder:
    """Deterministic hash-based embeddings with the Embedder interface.

    Vectors are unit length so inner product stays a cosine similarity, and
    identical text always produces an identical vector, which is all the
    retrieval and routing code actually requires of an embedder.
    """

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype="float32")
        rows = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            vector = np.random.default_rng(seed).normal(size=self._dimension)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype="float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@pytest.fixture(scope="session")
def config():
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def index_available(config) -> bool:
    from branched_rag.index.store import IndexStore

    return IndexStore.exists(config)
