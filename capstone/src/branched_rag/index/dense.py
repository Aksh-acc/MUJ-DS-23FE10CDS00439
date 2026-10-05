"""Exact dense retrieval over unit vectors.

``IndexFlatIP`` is a brute-force inner-product index. An approximate index (IVF,
HNSW) is the right choice at millions of vectors, but each branch here holds
single-digit thousands of chunks: a full scan is already sub-millisecond, and an
approximate index would add a training step, a tuning parameter and a recall
ceiling in exchange for nothing measurable. Because vectors arrive L2-normalised
from the embedder, inner product is exactly cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class DenseIndex:
    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> "DenseIndex":
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(index)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def vectors(self) -> np.ndarray:
        """Read the stored vectors back out.

        A flat index keeps them verbatim, so this is exact rather than an
        approximation. The evaluation harness uses it to build its pooled
        baseline from the same embeddings as the branched system, instead of
        re-encoding the corpus and introducing a second variable.
        """
        if self.size == 0:
            return np.zeros((0, self._index.d), dtype="float32")
        return self._index.reconstruct_n(0, self.size)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return ``(row, score)`` pairs ordered by descending cosine similarity."""
        if self.size == 0 or top_k <= 0:
            return []
        matrix = np.ascontiguousarray(query.reshape(1, -1), dtype="float32")
        scores, rows = self._index.search(matrix, min(top_k, self.size))
        return [
            (int(row), float(score))
            for row, score in zip(rows[0], scores[0])
            if row != -1
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        if not path.exists():
            raise FileNotFoundError(f"dense index not found: {path}")
        return cls(faiss.read_index(str(path)))
