"""Corpus construction: fetch, clean, chunk and persist the three branch corpora.

Produces two tables so the expensive stage can be cached independently of the
cheap one:

``corpus.parquet``
    One row per document, cleaned but unchunked. This is the auditable record of
    what the system was allowed to read.
``chunks.parquet``
    One row per retrievable unit. Chunk geometry is per-branch, so re-tuning
    ``chunk_size`` for one branch does not require re-downloading anything.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from ..config import Config
from .preprocess import chunk_text, clean_text
from .sources import load_source

logger = logging.getLogger(__name__)

CORPUS_COLUMNS = ["doc_id", "branch", "title", "text", "url", "source", "extra"]
CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "branch",
    "title",
    "text",
    "url",
    "source",
    "chunk_index",
    "n_chunks",
    "n_chars",
    "extra",
]


@dataclass
class IngestReport:
    documents_per_branch: dict[str, int]
    chunks_per_branch: dict[str, int]
    mean_chunk_chars: dict[str, float]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "branch": list(self.documents_per_branch),
                "documents": list(self.documents_per_branch.values()),
                "chunks": [self.chunks_per_branch[b] for b in self.documents_per_branch],
                "mean_chunk_chars": [
                    round(self.mean_chunk_chars[b], 1) for b in self.documents_per_branch
                ],
            }
        )


def build_corpus(config: Config) -> pd.DataFrame:
    """Download and clean every configured branch into one document table."""
    frames: list[pd.DataFrame] = []
    for name, branch in config.branches.items():
        logger.info("fetching branch %s from %s", name, branch.source)
        documents = load_source(branch.source, branch.documents, name)

        rows = []
        for document in documents:
            text = clean_text(document["text"])
            if not text:
                continue
            rows.append(
                {
                    "doc_id": document["doc_id"],
                    "branch": name,
                    "title": clean_text(document["title"]),
                    "text": text,
                    "url": document["url"],
                    "source": document["source"],
                    "extra": json.dumps(document["extra"], ensure_ascii=False),
                }
            )
        logger.info("branch %s: %d documents after cleaning", name, len(rows))
        frames.append(pd.DataFrame(rows, columns=CORPUS_COLUMNS))

    corpus = pd.concat(frames, ignore_index=True)
    config.paths.corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(config.paths.corpus, index=False)
    logger.info("wrote %d documents to %s", len(corpus), config.paths.corpus)
    return corpus


def build_chunks(config: Config, corpus: pd.DataFrame | None = None) -> pd.DataFrame:
    """Chunk a document table using each branch's own chunk geometry."""
    if corpus is None:
        corpus = load_corpus(config)

    rows = []
    for name, branch in config.branches.items():
        branch_docs = corpus[corpus["branch"] == name]
        for document in branch_docs.itertuples(index=False):
            pieces = chunk_text(document.text, branch.chunk_size, branch.chunk_overlap)
            for position, piece in enumerate(pieces):
                rows.append(
                    {
                        "chunk_id": f"{document.doc_id}#{position}",
                        "doc_id": document.doc_id,
                        "branch": name,
                        "title": document.title,
                        "text": piece,
                        "url": document.url,
                        "source": document.source,
                        "chunk_index": position,
                        "n_chunks": len(pieces),
                        "n_chars": len(piece),
                        "extra": document.extra,
                    }
                )

    chunks = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    config.paths.chunks.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(config.paths.chunks, index=False)
    logger.info("wrote %d chunks to %s", len(chunks), config.paths.chunks)
    return chunks


def load_corpus(config: Config) -> pd.DataFrame:
    if not config.paths.corpus.exists():
        raise FileNotFoundError(
            f"corpus not found at {config.paths.corpus}; run scripts/build_corpus.py first"
        )
    return pd.read_parquet(config.paths.corpus)


def load_chunks(config: Config) -> pd.DataFrame:
    if not config.paths.chunks.exists():
        raise FileNotFoundError(
            f"chunks not found at {config.paths.chunks}; run scripts/build_corpus.py first"
        )
    return pd.read_parquet(config.paths.chunks)


def summarise(chunks: pd.DataFrame) -> IngestReport:
    grouped = chunks.groupby("branch", sort=False)
    return IngestReport(
        documents_per_branch=grouped["doc_id"].nunique().to_dict(),
        chunks_per_branch=grouped.size().to_dict(),
        mean_chunk_chars=grouped["n_chars"].mean().to_dict(),
    )
