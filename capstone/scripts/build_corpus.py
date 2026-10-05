"""Fetch, clean and chunk the three branch corpora.

    python scripts/build_corpus.py
    python scripts/build_corpus.py --chunks-only    # re-chunk without re-downloading
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branched_rag import logging_setup
from branched_rag.config import load_config
from branched_rag.data import build_chunks, build_corpus, load_corpus, summarise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="reuse the existing corpus.parquet and only rebuild chunks",
    )
    args = parser.parse_args()

    logging_setup.configure()
    config = load_config(args.config)

    corpus = load_corpus(config) if args.chunks_only else build_corpus(config)
    chunks = build_chunks(config, corpus)

    print()
    print(summarise(chunks).as_frame().to_string(index=False))
    print(f"\ndocuments: {len(corpus)}   chunks: {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
