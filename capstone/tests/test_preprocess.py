"""Tests for cleaning and chunking.

The cleaning cases are the real defects measured in the source datasets, quoted
verbatim, so a regression here is caught as a corpus-quality bug rather than
discovered later as poor retrieval.
"""

from __future__ import annotations

import pytest

from branched_rag.data.preprocess import chunk_text, clean_text, split_sentences

BACKSLASH = chr(92)


class TestCleanText:
    def test_lone_backslash_becomes_a_space(self):
        # ag_news encodes an inter-word space as a backslash. Treating it as an
        # escape would delete the "b" and produce "dwindling and".
        raw = f"Wall Street's dwindling{BACKSLASH}band of ultra-cynics"
        assert clean_text(raw) == "Wall Street's dwindling band of ultra-cynics"

    def test_escaped_dollar_is_unescaped_not_spaced(self):
        raw = f"a record {BACKSLASH}$55.8bn as oil costs rose"
        assert clean_text(raw) == "a record $55.8bn as oil costs rose"

    def test_entity_fragment_missing_ampersand_is_repaired(self):
        assert clean_text("surged past #36;46 a barrel") == "surged past $46 a barrel"
        assert clean_text("It isn#39;t clear") == "It isn't clear"

    def test_standard_html_entities_are_unescaped(self):
        assert clean_text("profit &amp; loss &lt;eom&gt;") == "profit & loss <eom>"

    def test_escaped_quotes_are_unescaped(self):
        raw = f'He said {BACKSLASH}"it was fine{BACKSLASH}"'
        assert clean_text(raw) == 'He said "it was fine"'

    def test_runs_of_punctuation_are_collapsed(self):
        assert clean_text("Buyer Beware!!!!") == "Buyer Beware!"

    def test_whitespace_is_collapsed(self):
        assert clean_text("  too   many\n\tspaces ") == "too many spaces"

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_input_is_safe(self, value):
        assert clean_text(value) == ""


class TestSplitSentences:
    def test_splits_on_terminators(self):
        assert split_sentences("One thing. Two things! Three?") == [
            "One thing.",
            "Two things!",
            "Three?",
        ]

    def test_decimal_figures_are_not_split(self):
        # Dense in the financials branch; splitting here would halve a figure.
        assert split_sentences("Revenue rose to EUR 47.5 million in Q3.") == [
            "Revenue rose to EUR 47.5 million in Q3."
        ]

    def test_abbreviations_are_not_split(self):
        assert split_sentences("Mr. Smith of Acme Inc. agreed.") == [
            "Mr. Smith of Acme Inc. agreed."
        ]

    def test_initials_are_not_split(self):
        assert split_sentences("J. Smith reported it.") == ["J. Smith reported it."]

    def test_us_abbreviation_is_not_split(self):
        assert split_sentences("The U.S. unit grew 8%.") == ["The U.S. unit grew 8%."]

    def test_empty_input_returns_empty_list(self):
        assert split_sentences("") == []


class TestChunkText:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("One sentence only.", 200, 20) == ["One sentence only."]

    def test_respects_the_size_budget(self):
        text = " ".join(f"Sentence number {i} here." for i in range(40))
        chunks = chunk_text(text, 120, 0)
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)

    def test_no_chunk_starts_mid_sentence(self):
        text = " ".join(f"Sentence number {i} here." for i in range(20))
        for chunk in chunk_text(text, 100, 30):
            assert chunk[0].isupper()

    def test_overlap_repeats_whole_sentences(self):
        text = "A one. B two. C three. D four. E five."
        chunks = chunk_text(text, 24, 10)
        assert len(chunks) > 1
        # The tail of one chunk must reappear at the head of the next.
        first_tail = chunks[0].split(". ")[-1]
        assert chunks[1].startswith(first_tail.rstrip("."))

    def test_oversized_sentence_is_split_at_word_boundaries(self):
        # Spans with no usable sentence boundary (press-release contact blocks)
        # once produced a 13,066-character chunk. They are now split on
        # whitespace so the budget holds.
        long_sentence = "word " * 80 + "end."
        chunks = chunk_text(long_sentence, 50, 10)
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
        assert all(" word" not in chunk[:1] for chunk in chunks)

    def test_contact_block_without_sentence_boundaries_stays_in_budget(self):
        block = (
            "Contacts For investor inquiries contact Ms. Cara O'Brien FTI "
            "Consulting Tel: +852-3768-4537 Email: cara.obrien@example.com "
        ) * 6
        chunks = chunk_text(block, 640, 80)
        assert len(chunks) > 1
        assert max(len(chunk) for chunk in chunks) <= 640

    def test_single_token_longer_than_budget_is_not_truncated(self):
        # Truncating would destroy a URL or identifier without making the chunk
        # usable, so the token is kept intact and the budget knowingly exceeded.
        token = "x" * 900
        chunks = chunk_text(token + ".", 100, 10)
        assert len(chunks) == 1
        assert token in chunks[0]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("", 100, 10) == []

    @pytest.mark.parametrize(
        "size,overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)]
    )
    def test_invalid_geometry_is_rejected(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text("Some text here.", size, overlap)
