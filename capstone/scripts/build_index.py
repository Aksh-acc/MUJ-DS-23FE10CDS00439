"""Embed every branch corpus and persist the per-branch dense indexes.

    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from branched_rag import logging_setup
from branched_rag.config import load_config
from branched_rag.index import IndexStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    logging_setup.configure()
    config = load_config(args.config)

    started = time.perf_counter()
    store = IndexStore.build(config)
    elapsed = time.perf_counter() - started

    print()
    print(store.stats().to_string(index=False))
    print(f"\n{store.total_chunks} chunks indexed in {elapsed:.1f}s")
    print(f"artifacts: {config.paths.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
