"""Tests for the evaluation metrics."""

from __future__ import annotations

import pytest

from branched_rag.evaluation.metrics import (
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    set_score,
    source_coverage,
)


class TestSetScore:
    def test_exact_agreement(self):
        score = set_score({"a", "b"}, {"a", "b"})
        assert score.exact_match
        assert score.precision == score.recall == score.f1 == 1.0

    def test_over_selection_costs_precision_only(self):
        score = set_score({"a", "b"}, {"a"})
        assert score.recall == 1.0
        assert score.precision == 0.5
        assert not score.exact_match

    def test_under_selection_costs_recall_only(self):
        score = set_score({"a"}, {"a", "b"})
        assert score.precision == 1.0
        assert score.recall == 0.5

    def test_disjoint_sets_score_zero(self):
        score = set_score({"c"}, {"a", "b"})
        assert score.f1 == 0.0

    def test_empty_prediction_scores_zero(self):
        assert set_score(set(), {"a"}).f1 == 0.0

    def test_empty_expected_is_rejected(self):
        with pytest.raises(ValueError):
            set_score({"a"}, set())


class TestRankMetrics:
    def test_recall_at_k_respects_the_cutoff(self):
        retrieved = ["a", "b", "c", "d"]
        assert recall_at_k(retrieved, {"a", "d"}, k=2) == 0.5
        assert recall_at_k(retrieved, {"a", "d"}, k=4) == 1.0

    def test_reciprocal_rank_uses_the_first_hit(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
        assert reciprocal_rank(["a", "y"], {"a"}) == 1.0

    def test_reciprocal_rank_is_zero_on_a_miss(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_ndcg_rewards_a_higher_rank(self):
        early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        assert early > late
        assert early == 1.0

    def test_ndcg_is_bounded(self):
        value = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert value == pytest.approx(1.0)

    def test_metrics_handle_an_empty_relevant_set(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0
        assert ndcg_at_k(["a"], set(), k=1) == 0.0


class TestSourceCoverage:
    def test_full_coverage(self):
        assert source_coverage(["reviews", "news"], {"reviews", "news"}) == 1.0

    def test_partial_coverage_when_one_source_is_missing(self):
        # The flat-index failure mode: plenty of evidence, all from one branch.
        assert source_coverage(
            ["financials"] * 10, {"reviews", "financials"}
        ) == 0.5

    def test_extra_sources_do_not_inflate_coverage(self):
        assert source_coverage(["a", "b", "reviews"], {"reviews"}) == 1.0

    def test_empty_expected_scores_zero(self):
        assert source_coverage(["reviews"], set()) == 0.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0]) == 1.5
