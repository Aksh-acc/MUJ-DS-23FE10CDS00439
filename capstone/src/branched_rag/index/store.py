"""Per-branch index construction, persistence and loading.

The central structural decision of the project lives here: there is no single
index. Each branch owns a dense index, a sparse index and its own chunk table,
and nothing is ever merged.

The measured reason is corpus imbalance. The three branches contribute very
unequal chunk counts because their documents differ in length -- long Reuters
articles chunk into many more pieces than a short customer review. A flat index
lets the largest branch win top-k on volume alone, so the retriever returns
whatever source is most verbose rather than whatever source answers the
question. Separate indexes make per-branch budgets explicit: the evidence mix is
set by the router, not by an accident of document length.

Separation also lets each branch keep its own dense/sparse balance, which the
flat design cannot express at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BranchConfig, Config
from ..data.ingest import load_chunks
from .dense import DenseIndex
from .embedder import Embedder
from .sparse import SparseIndex

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass
class BranchIndex:
    """Everything needed to retrieve from one branch."""

    config: BranchConfig
    chunks: pd.DataFrame
    dense: DenseIndex
    sparse: SparseIndex

    def __post_init__(self) -> None:
        if len(self.chunks) != self.dense.size:
            raise ValueError(
                f"branch {self.config.name}: {len(self.chunks)} chunks but "
                f"{self.dense.size} dense vectors; rebuild the index"
            )

    @property
    def size(self) -> int:
        return len(self.chunks)

    def record(self, row: int) -> dict:
        """Return one chunk as a plain dict, with ``extra`` decoded."""
        chunk = self.chunks.iloc[row]
        try:
            extra = json.loads(chunk["extra"]) if chunk["extra"] else {}
        except (TypeError, ValueError):
            extra = {}
        return {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "branch": chunk["branch"],
            "title": chunk["title"],
            "text": chunk["text"],
            "url": chunk["url"],
            "source": chunk["source"],
            "extra": extra,
        }


class IndexStore:
    """Holds one BranchIndex per configured branch."""

    def __init__(self, config: Config, branches: dict[str, BranchIndex]) -> None:
        self.config = config
        self.branches = branches

    def __getitem__(self, name: str) -> BranchIndex:
        try:
            return self.branches[name]
        except KeyError:
            raise KeyError(
                f"branch {name!r} is not indexed; built branches: {sorted(self.branches)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.branches

    @property
    def names(self) -> list[str]:
        return list(self.branches)

    @property
    def total_chunks(self) -> int:
        return sum(index.size for index in self.branches.values())

    def stats(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "branch": name,
                    "label": index.config.label,
                    "chunks": index.size,
                    "share_of_corpus": round(index.size / max(self.total_chunks, 1), 3),
                    "top_k": index.config.top_k,
                    "dense_weight": index.config.dense_weight,
                    "sparse_weight": index.config.sparse_weight,
                }
                for name, index in self.branches.items()
            ]
        )

    @classmethod
    def build(cls, config: Config, chunks: pd.DataFrame | None = None) -> "IndexStore":
        if chunks is None:
            chunks = load_chunks(config)

        embedder = Embedder(config.embedding)
        artifacts = config.paths.artifacts
        artifacts.mkdir(parents=True, exist_ok=True)

        branches: dict[str, BranchIndex] = {}
        manifest = {
            "embedding_model": config.embedding.model,
            "dimension": embedder.dimension,
            "branches": {},
        }

        for name, branch in config.branches.items():
            branch_chunks = (
                chunks[chunks["branch"] == name].reset_index(drop=True).copy()
            )
            if branch_chunks.empty:
                logger.warning("branch %s has no chunks; skipping", name)
                continue

            texts = branch_chunks["text"].tolist()
            logger.info("embedding %d chunks for branch %s", len(texts), name)
            vectors = embedder.encode(texts, show_progress=True)

            dense = DenseIndex.build(vectors)
            dense.save(artifacts / f"{name}.faiss")
            branch_chunks.to_parquet(artifacts / f"{name}.parquet", index=False)

            branches[name] = BranchIndex(
                config=branch,
                chunks=branch_chunks,
                dense=dense,
                sparse=SparseIndex.build(texts),
            )
            manifest["branches"][name] = {"chunks": len(texts)}

        (artifacts / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info("built %d branch indexes in %s", len(branches), artifacts)
        return cls(config, branches)

    @classmethod
    def load(cls, config: Config) -> "IndexStore":
        artifacts = config.paths.artifacts
        manifest_path = artifacts / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no index manifest at {manifest_path}; run scripts/build_index.py first"
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != config.embedding.model:
            raise ValueError(
                f"index was built with {manifest.get('embedding_model')!r} but config "
                f"requests {config.embedding.model!r}; rebuild the index"
            )

        branches: dict[str, BranchIndex] = {}
        for name in manifest.get("branches", {}):
            if name not in config.branches:
                logger.warning("index contains unconfigured branch %s; ignoring", name)
                continue
            branch_chunks = pd.read_parquet(artifacts / f"{name}.parquet")
            branches[name] = BranchIndex(
                config=config.branch(name),
                chunks=branch_chunks,
                dense=DenseIndex.load(artifacts / f"{name}.faiss"),
                # Rebuilt rather than unpickled; see sparse.py for why.
                sparse=SparseIndex.build(branch_chunks["text"].tolist()),
            )
        logger.info("loaded %d branch indexes", len(branches))
        return cls(config, branches)

    @staticmethod
    def exists(config: Config) -> bool:
        return (config.paths.artifacts / MANIFEST_NAME).exists()


def branch_descriptor_matrix(
    config: Config, embedder: Embedder
) -> tuple[list[str], np.ndarray]:
    """Embed each branch's descriptor text, for use by the router."""
    names = config.branch_names
    descriptors = [config.branch(name).descriptor for name in names]
    return names, embedder.encode(descriptors)
