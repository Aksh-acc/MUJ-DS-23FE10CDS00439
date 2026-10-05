"""Integration tests against the built index.

These skip rather than fail when no index is present, so a clean checkout can
run the suite before `build_index.py` has been executed. Generation is switched
off throughout: these assert the retrieval contract, not model output, so they
need no API key and spend no tokens.
"""

from __future__ import annotations

import pytest

from branched_rag.config import load_config
from branched_rag.data.ingest import load_chunks
from branched_rag.index.store import IndexStore
from branched_rag.pipeline import BranchedRAGPipeline

pytestmark = pytest.mark.skipif(
    not IndexStore.exists(load_config()),
    reason="no index built; run scripts/build_corpus.py and scripts/build_index.py",
)


@pytest.fixture(scope="module")
def pipeline() -> BranchedRAGPipeline:
    return BranchedRAGPipeline.load()


class TestIndexIntegrity:
    def test_every_configured_branch_is_indexed(self, pipeline):
        assert set(pipeline.store.names) == set(pipeline.config.branch_names)

    def test_chunk_counts_match_the_dense_vectors(self, pipeline):
        for name in pipeline.store.names:
            index = pipeline.store[name]
            assert index.size == index.dense.size

    def test_branch_shares_sum_to_one(self, pipeline):
        assert pipeline.store.stats()["share_of_corpus"].sum() == pytest.approx(
            1.0, abs=0.01
        )

    def test_corpus_has_no_leftover_escape_artifacts(self):
        # Guards the cleaning pipeline against a source-format change.
        chunks = load_chunks(load_config())
        offenders = chunks["text"].str.contains(
            r"\\|#\d+;|&amp;|&#", regex=True, na=False
        )
        assert offenders.sum() == 0

    def test_reviews_branch_is_polarity_balanced(self):
        import json

        chunks = load_chunks(load_config())
        reviews = chunks[chunks["branch"] == "reviews"].drop_duplicates("doc_id")
        labels = reviews["extra"].map(
            lambda value: json.loads(value).get("stated_polarity")
        )
        counts = labels.value_counts(normalize=True)
        # Balanced at ingest so sentiment aggregates measure the query, not the
        # corpus. Allow a small margin for documents dropped during cleaning.
        assert abs(counts.get("positive", 0) - 0.5) < 0.05


class TestRetrievalBehaviour:
    def test_review_question_routes_to_reviews(self, pipeline):
        answer = pipeline.answer(
            "What do customers complain about in chargers?", generate=False
        )
        assert "reviews" in answer.plan.selected

    def test_financial_question_routes_to_financials(self, pipeline):
        answer = pipeline.answer(
            "What were the reported quarterly revenue and profit figures?",
            generate=False,
        )
        assert "financials" in answer.plan.selected

    def test_compound_question_selects_several_branches(self, pipeline):
        # The clause-splitting fix exists for exactly this case; a single-vector
        # router collapsed it to one branch.
        answer = pipeline.answer(
            "Should we launch a budget tablet? Consider demand, rivals and "
            "financial risk.",
            generate=False,
        )
        assert len(answer.plan.selected) >= 2

    def test_evidence_respects_each_branch_top_k(self, pipeline):
        answer = pipeline.answer("revenue growth and customer feedback", generate=False)
        for result in answer.results:
            assert len(result.evidence) <= pipeline.config.branch(result.branch).top_k

    def test_citation_numbers_are_unique_and_contiguous(self, pipeline):
        answer = pipeline.answer("market trends and earnings", generate=False)
        numbers = [item.citation for item in answer.evidence]
        assert len(numbers) == len(set(numbers))
        assert numbers == list(range(1, len(numbers) + 1))

    def test_evidence_only_comes_from_selected_branches(self, pipeline):
        answer = pipeline.answer("quarterly earnings guidance", generate=False)
        assert {item.branch for item in answer.evidence} <= set(answer.plan.selected)

    def test_manual_override_replaces_the_router_choice(self, pipeline):
        answer = pipeline.answer(
            "quarterly earnings guidance", branches=["reviews"], generate=False
        )
        assert answer.plan.selected == ["reviews"]
        assert {item.branch for item in answer.evidence} == {"reviews"}

    def test_every_passage_carries_a_retrieval_path(self, pipeline):
        answer = pipeline.answer("battery complaints", generate=False)
        for item in answer.evidence:
            assert item.retrieval_path in {
                "dense + lexical",
                "dense only",
                "lexical only",
            }

    def test_timings_are_recorded_for_each_stage(self, pipeline):
        answer = pipeline.answer("margin pressure", generate=False)
        assert {"route", "retrieve"} <= set(answer.timings)
        assert all(value >= 0 for value in answer.timings.values())

    def test_empty_query_is_rejected(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.answer("   ", generate=False)


class TestFlatBaselineComparison:
    def test_branched_covers_more_sources_than_flat_on_multi_source_queries(
        self, pipeline
    ):
        """The project's central claim, asserted as a test.

        A pooled index biases toward whichever branch contributes most chunks.
        Branched retrieval allocates evidence per branch instead, so on queries
        that genuinely need several sources it must not cover fewer of them.
        """
        from branched_rag.evaluation import (
            compare_flat_and_branched,
            load_eval_queries,
        )

        queries = [
            query
            for query in load_eval_queries(pipeline.config.paths.eval_queries)
            if query.kind == "multi"
        ]
        _, summary = compare_flat_and_branched(pipeline, queries)
        row = summary.iloc[0]
        assert row["branched_coverage"] > row["flat_coverage"]
        assert row["branched_sources"] > row["flat_sources"]
