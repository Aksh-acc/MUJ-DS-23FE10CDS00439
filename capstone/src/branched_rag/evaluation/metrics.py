"""Metric implementations, kept free of pipeline imports so they stay testable.

Set metrics are used for routing, where the prediction is a set of branches, and
rank metrics for retrieval, where the prediction is an ordered list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SetScore:
    precision: float
    recall: float
    f1: float
    exact_match: bool


def set_score(predicted: set[str], expected: set[str]) -> SetScore:
    """Precision, recall and F1 over a predicted set, plus exact-set agreement.

    Exact match is reported alongside F1 because they answer different questions
    for a router. F1 rewards getting most of a multi-branch answer right; exact
    match asks whether the branch plan was fully correct, which is what actually
    determines the evidence the user sees.
    """
    if not expected:
        raise ValueError("expected set must not be empty")
    hits = len(predicted & expected)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(expected)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return SetScore(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=predicted == expected,
    )


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of relevant items appearing in the first ``k`` retrieved."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant item, or 0 if none was retrieved."""
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k.

    With binary relevance the discount is what distinguishes this from recall:
    a relevant item at rank 1 is worth more than the same item at rank 5, which
    is the property that makes nDCG sensitive to reranking.
    """
    if not relevant:
        return 0.0
    gains = [
        1.0 / math.log2(position + 1)
        for position, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    ]
    dcg = sum(gains)
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def source_coverage(branches: list[str], expected: set[str]) -> float:
    """Share of the expected branches that contributed any retrieved evidence.

    This is the metric that tests the project's central claim. A flat index can
    score well on relevance while returning evidence from only the most verbose
    source; coverage measures whether a multi-source question actually received
    multi-source evidence.
    """
    if not expected:
        return 0.0
    return len(set(branches) & expected) / len(expected)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
