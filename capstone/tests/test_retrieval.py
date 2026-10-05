"""Tests for rank fusion, tokenisation, routing and citation handling."""

from __future__ import annotations

import pytest

from branched_rag.config import BranchConfig
from branched_rag.generation.synthesis import (
    extract_citations,
    normalize_citation_markers,
)
from branched_rag.index.fusion import reciprocal_rank_fusion
from branched_rag.index.sparse import SparseIndex, tokenize
from branched_rag.retrieval.query_ops import QueryExpander, normalize_query
from branched_rag.retrieval.router import BranchRouter, split_clauses


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_is_rewarded(self):
        # Row 1 is second in both lists; row 0 is first in one and absent from the
        # other. Appearing in both should let row 1 overtake it.
        fused = reciprocal_rank_fusion(
            dense=[(0, 0.9), (1, 0.8)],
            sparse=[(2, 12.0), (1, 11.0)],
            dense_weight=0.5,
            sparse_weight=0.5,
            k=1,
        )
        assert fused[0].row == 1
        assert fused[0].found_by_both

    def test_raw_scores_do_not_affect_the_outcome(self):
        # BM25 scores are unbounded; only rank may matter.
        small = reciprocal_rank_fusion([(0, 0.1)], [(1, 0.2)], k=60)
        large = reciprocal_rank_fusion([(0, 0.1)], [(1, 9999.0)], k=60)
        assert [hit.row for hit in small] == [hit.row for hit in large]

    def test_weights_shift_the_ranking(self):
        dense = [(0, 0.9)]
        sparse = [(1, 5.0)]
        dense_heavy = reciprocal_rank_fusion(dense, sparse, 0.9, 0.1, k=60)
        sparse_heavy = reciprocal_rank_fusion(dense, sparse, 0.1, 0.9, k=60)
        assert dense_heavy[0].row == 0
        assert sparse_heavy[0].row == 1

    def test_ranks_record_which_retriever_found_each_row(self):
        fused = reciprocal_rank_fusion([(7, 0.5)], [(8, 1.0)], k=60)
        by_row = {hit.row: hit for hit in fused}
        assert by_row[7].dense_rank == 1 and by_row[7].sparse_rank is None
        assert by_row[8].sparse_rank == 1 and by_row[8].dense_rank is None

    def test_ties_are_broken_deterministically(self):
        first = reciprocal_rank_fusion([(5, 1.0), (3, 1.0)], [], k=60, top_k=2)
        second = reciprocal_rank_fusion([(5, 1.0), (3, 1.0)], [], k=60, top_k=2)
        assert [h.row for h in first] == [h.row for h in second]

    def test_empty_inputs_yield_no_hits(self):
        assert reciprocal_rank_fusion([], [], k=60) == []

    def test_non_positive_k_is_rejected(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([(0, 1.0)], [], k=0)


class TestTokenize:
    def test_figures_and_percentages_survive_as_single_tokens(self):
        tokens = tokenize("Revenue rose 8% to EUR 47.5 million in Q3")
        assert "47.5" in tokens
        assert "8%" in tokens
        assert "q3" in tokens

    def test_stopwords_are_dropped(self):
        assert "the" not in tokenize("the revenue of the company")

    def test_negations_are_kept(self):
        # "not" carries the signal in a complaint or a profit warning.
        assert "not" in tokenize("the charger does not work")

    def test_single_characters_are_dropped(self):
        assert tokenize("a b revenue") == ["revenue"]


class TestSparseIndex:
    def test_matches_a_rare_term(self):
        index = SparseIndex.build(
            [
                "the battery charger failed after two months",
                "quarterly revenue rose to EUR 47.5 million",
                "a new tablet was announced today",
            ]
        )
        hits = index.search("EUR 47.5 million revenue", top_k=2)
        assert hits and hits[0][0] == 1

    def test_empty_corpus_is_safe(self):
        assert SparseIndex.build([]).search("anything", top_k=5) == []

    def test_query_of_only_stopwords_returns_nothing(self):
        index = SparseIndex.build(["some real content here"])
        assert index.search("the of and", top_k=5) == []


class TestSplitClauses:
    def test_splits_a_compound_question(self):
        clauses = split_clauses(
            "Should we launch a budget tablet? Consider demand, rivals and financial risk."
        )
        assert "financial risk" in clauses
        assert any("budget tablet" in clause for clause in clauses)

    def test_simple_query_is_not_fragmented(self):
        assert split_clauses("What do customers think") == ["What do customers think"]

    def test_empty_fragments_are_dropped(self):
        assert split_clauses("a,,, b.") == ["a", "b"]


class TestBranchRouter:
    def test_lexical_strategy_selects_on_keyword_overlap(self, config, fake_embedder):
        router = BranchRouter(config, fake_embedder)
        plan = router.route("revenue profit margin earnings", strategy="lexical")
        assert "financials" in plan.selected

    def test_never_returns_an_empty_plan(self, config, fake_embedder):
        router = BranchRouter(config, fake_embedder)
        plan = router.route("zzzz qqqq", strategy="hybrid")
        assert len(plan.selected) >= 1

    def test_respects_max_branches(self, config, fake_embedder):
        router = BranchRouter(config, fake_embedder)
        plan = router.route(
            "revenue profit customer complaint competitor launch market",
            strategy="hybrid",
        )
        assert len(plan.selected) <= config.router.max_branches

    def test_unknown_strategy_is_rejected(self, config, fake_embedder):
        router = BranchRouter(config, fake_embedder)
        with pytest.raises(ValueError):
            router.route("anything", strategy="telepathy")

    def test_decisions_cover_every_branch_and_are_score_ordered(
        self, config, fake_embedder
    ):
        router = BranchRouter(config, fake_embedder)
        plan = router.route("revenue and customer reviews", strategy="hybrid")
        assert {d.branch for d in plan.decisions} == set(config.branch_names)
        scores = [d.score for d in plan.decisions]
        assert scores == sorted(scores, reverse=True)


class TestQueryExpander:
    @staticmethod
    def _branch() -> BranchConfig:
        return BranchConfig(
            name="financials",
            label="Financial Reporting",
            source="x",
            description="revenue and profit reporting",
            keywords=("revenue", "cost", "costs", "margin", "margins", "profit"),
            documents=1,
            chunk_size=100,
            chunk_overlap=0,
            top_k=3,
            dense_weight=0.5,
            sparse_weight=0.5,
        )

    def test_expansion_skips_morphological_duplicates(self, fake_embedder):
        expander = QueryExpander(fake_embedder)
        result = expander.expand("how did the company perform", self._branch())
        added = result.added_terms
        assert len(added) == len(set(added))
        # "cost"/"costs" and "margin"/"margins" must not both be added.
        assert not any(
            a != b and (a.startswith(b) or b.startswith(a))
            for a in added
            for b in added
        )

    def test_terms_already_in_the_query_are_not_repeated(self, fake_embedder):
        expander = QueryExpander(fake_embedder)
        result = expander.expand("revenue", self._branch())
        assert "revenue" not in result.added_terms

    def test_expanded_text_contains_the_original_query(self, fake_embedder):
        expander = QueryExpander(fake_embedder)
        result = expander.expand("what about profit", self._branch())
        assert result.text.startswith("what about profit")


class TestCitations:
    def test_plain_and_grouped_forms_are_parsed(self):
        assert extract_citations("a [3] b [7, 9] c [11][12]") == {3, 7, 9, 11, 12}

    def test_full_width_browser_markers_are_normalised(self):
        # gpt-oss emits this form non-deterministically; unhandled it reads as
        # zero citations and the grounding metric silently collapses.
        text = "Reuters notes X 【5†L0-L2】 and 【6】"
        assert normalize_citation_markers(text) == "Reuters notes X [5] and [6]"
        assert extract_citations(text) == {5, 6}

    def test_bracketed_line_reference_form_is_normalised(self):
        assert extract_citations("see [5†L1-L4]") == {5}

    def test_non_numeric_brackets_are_ignored(self):
        assert extract_citations("see [source] and [n]") == set()

    def test_empty_text_yields_no_citations(self):
        assert extract_citations("") == set()
        assert extract_citations(None) == set()


def test_normalize_query_collapses_whitespace():
    assert normalize_query("  a   b \n c ") == "a b c"
