"""Single place that configures logging for the CLI scripts.

The Streamlit app does not call this; Streamlit owns its own root handler and
adding another duplicates every line in the terminal.
"""

from __future__ import annotations

import logging


def configure(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "filelock", "httpx", "datasets", "fsspec", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
