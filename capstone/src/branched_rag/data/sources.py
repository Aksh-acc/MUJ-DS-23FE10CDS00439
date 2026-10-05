"""Adapters that turn public Hugging Face datasets into a common document schema.

Each adapter yields dicts with the keys used everywhere downstream:
``doc_id``, ``branch``, ``title``, ``text``, ``url``, ``source``, ``extra``.

Datasets are read in streaming mode and consumed in file order, so a given
``limit`` always produces the same corpus. That keeps the pipeline reproducible
without committing gigabytes of raw data to the repository.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from datasets import load_dataset

Document = dict[str, Any]

# amazon_polarity mixes physical goods with books, music and films. Market
# intelligence questions concern things that get manufactured, priced and
# shipped, so a review is admitted only if it looks like one.
#
# Cues are matched on word boundaries, not as substrings. Substring matching
# admitted a film review because "motor" occurs inside "motorcycle", and that
# single leak is why the rule is a compiled alternation rather than `in`.
#
# Two distinct cues are required rather than one. The filter draws from 3.6M
# candidate rows to fill 2000 slots, so precision is nearly free and recall is
# irrelevant. The rule stays lexical and visible so it can be defended in a
# report, rather than delegating corpus admission to an opaque classifier.
_PRODUCT_CUES = re.compile(
    r"\b("
    r"batter(?:y|ies)|charger|charging|headphones?|earphones?|earbuds?|speakers?|"
    r"bluetooth|wireless|usb|hdmi|adapter|cables?|laptops?|notebooks?|computers?|"
    r"monitors?|keyboards?|printers?|scanners?|cameras?|lenses?|tripods?|"
    r"phones?|tablets?|routers?|modems?|hard ?drives?|memory cards?|"
    r"appliances?|vacuums?|blenders?|kettles?|toasters?|microwaves?|"
    r"warrant(?:y|ies)|refunds?|returns?|build quality|"
    r"shipping|packaging|unboxing|installed|installation|instructions|manual|"
    r"bought this|purchased this|ordered this|this product|this item|this unit"
    r")\b"
)
# Valence-carrying terms are deliberately absent from the cue list. An earlier
# version included "defective", "flimsy" and "overpriced", and the resulting
# corpus was 67% negative: the filter was selecting complaints rather than
# product reviews, which would have biased every sentiment aggregate the reviews
# branch produces. Cues are now product nouns and neutral commerce terms only,
# and the sample is additionally balanced by stated polarity below.

# If any of these appear the row is dropped outright, whatever the product cues say.
_MEDIA_CUES = re.compile(
    r"\b("
    r"books?|novels?|authors?|writers?|chapters?|paperback|hardcover|"
    r"albums?|songs?|tracks?|lyrics?|cds?|soundtracks?|bands?|"
    r"movies?|films?|dvds?|blu-?rays?|actors?|actress|directors?|cast|"
    r"records?|vinyl|compilations?|"
    r"plot|characters?|storyline|episodes?|seasons?|sequel|"
    r"read it|reading it|listened to|watched it"
    r")\b"
)

_AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
_AG_NEWS_KEEP = frozenset({2, 3})  # Business and Sci/Tech only
_POLARITY_LABELS = {0: "negative", 1: "positive"}

_MIN_PRODUCT_CUES = 2


def _strip(value: Any) -> str:
    return str(value or "").strip()


def _is_product_review(text: str) -> bool:
    lowered = text.lower()
    if _MEDIA_CUES.search(lowered):
        return False
    return len(set(_PRODUCT_CUES.findall(lowered))) >= _MIN_PRODUCT_CUES


def load_amazon_polarity(limit: int, branch: str) -> Iterator[Document]:
    """Stream product reviews, balanced across the dataset's polarity label.

    The quota is split evenly between positive and negative reviews. Without
    this the branch inherits whatever mix the lexical filter happens to admit,
    and any sentiment aggregate computed over it measures the filter rather than
    the market. Balancing makes the aggregate interpretable: a branch that comes
    back 70% negative on a query is telling us something about the retrieved
    subset, not about corpus construction.
    """
    stream = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
    quota = {0: limit // 2, 1: limit - limit // 2}
    taken = {0: 0, 1: 0}
    kept = 0

    for row in stream:
        if kept >= limit:
            return
        label = row.get("label")
        if label not in quota or taken[label] >= quota[label]:
            continue
        body = _strip(row.get("content"))
        if len(body) < 180 or not _is_product_review(body):
            continue
        yield {
            "doc_id": f"{branch}-{kept:05d}",
            "branch": branch,
            "title": _strip(row.get("title")) or "Customer review",
            "text": body,
            "url": "",
            "source": "fancyzhx/amazon_polarity",
            "extra": {"stated_polarity": _POLARITY_LABELS[label]},
        }
        taken[label] += 1
        kept += 1


def load_ag_news(limit: int, branch: str) -> Iterator[Document]:
    stream = load_dataset("fancyzhx/ag_news", split="train", streaming=True)
    kept = 0
    for row in stream:
        if kept >= limit:
            return
        if row.get("label") not in _AG_NEWS_KEEP:
            continue
        body = _strip(row.get("text"))
        if len(body) < 120:
            continue
        # ag_news packs headline and lede into one field separated by " - ".
        head, _, rest = body.partition(" - ")
        if rest and len(head) < 140:
            title, text = head, rest
        else:
            title, text = "Market report", body
        yield {
            "doc_id": f"{branch}-{kept:05d}",
            "branch": branch,
            "title": title.strip(),
            "text": text.strip() or body,
            "url": "",
            "source": "fancyzhx/ag_news",
            "extra": {"category": _AG_NEWS_LABELS[row["label"]]},
        }
        kept += 1


def load_financial_news(limit: int, branch: str) -> Iterator[Document]:
    stream = load_dataset("ashraq/financial-news-articles", split="train", streaming=True)
    kept = 0
    for row in stream:
        if kept >= limit:
            return
        body = _strip(row.get("text"))
        if len(body) < 400:
            continue
        yield {
            "doc_id": f"{branch}-{kept:05d}",
            "branch": branch,
            "title": _strip(row.get("title")) or "Financial report",
            "text": body,
            "url": _strip(row.get("url")),
            "source": "ashraq/financial-news-articles",
            "extra": {},
        }
        kept += 1


SOURCE_REGISTRY: dict[str, Callable[[int, str], Iterator[Document]]] = {
    "amazon_polarity": load_amazon_polarity,
    "ag_news": load_ag_news,
    "financial_news": load_financial_news,
}


def load_source(source: str, limit: int, branch: str) -> list[Document]:
    try:
        loader = SOURCE_REGISTRY[source]
    except KeyError:
        raise KeyError(
            f"unknown source {source!r}; registered: {sorted(SOURCE_REGISTRY)}"
        ) from None
    return list(loader(limit, branch))
