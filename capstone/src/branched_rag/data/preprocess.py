r"""Text normalisation and sentence-aware chunking.

The cleaning rules are driven by defects measured in the three source datasets
rather than by a generic checklist. Each rule was added after inspecting real
rows; the ordering matters, which is why this is a pipeline and not one regex.

``ag_news`` was scraped in a way that lost two things:

* Spaces between words became single backslashes, so a lede reads
  ``Wall Street's dwindling\band of ultra-cynics``. Decoding ``\b`` as an escape
  would produce "dwindling and" and silently delete a word, so genuinely escaped
  characters (``\$``, ``\"``) are unescaped first and only then is a surviving
  lone backslash restored to a space.
* HTML entities lost their leading ampersand, leaving fragments such as
  ``#36;46`` for "$46" and ``isn#39;t`` for "isn't". These are repaired before
  unescaping, otherwise the digits survive into the index as nonsense tokens.

``amazon_polarity`` keeps escaped quotes inside review bodies, and all three
sources carry ordinary HTML entities from their scraped origin.

Chunking packs whole sentences up to a character budget instead of slicing at a
fixed stride. A chunk cut mid-clause hands the embedder a truncated proposition,
and evidence shown in the UI has to be quotable.
"""

from __future__ import annotations

import html
import re
import unicodedata

# Characters that were legitimately backslash-escaped in the source text.
_ESCAPED_LITERALS = re.compile(r"\\([$\"'/`*_\[\]()#])")

# Any surviving backslash stood in for a space (see module docstring).
_STRAY_BACKSLASH = re.compile(r"\\+")

# Entity fragments whose leading "&" was stripped during scraping.
_NUMERIC_ENTITY_FRAGMENT = re.compile(r"(?<!&)#(\d{2,4});")
_NAMED_ENTITY_FRAGMENT = re.compile(
    r"(?<!&)(quot|amp|lt|gt|nbsp|apos|ldquo|rdquo|rsquo|lsquo|mdash|ndash);"
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_REPEATED_PUNCT = re.compile(r"([!?.,;:])\1{2,}")
_WHITESPACE = re.compile(r"\s+")

# Candidate sentence boundaries: .!? then whitespace then the start of something
# new. Candidates are filtered by _is_real_boundary, because a fixed-width regex
# lookbehind cannot express "unless the preceding token is an abbreviation", and
# that exclusion is what stops "U.S. unit" and "EUR 47.5 million" being torn in
# half. Both forms are dense in the financial branch.
_BOUNDARY = re.compile(r"""[.!?]["')\]]*\s+(?=["'(\[]*[A-Z0-9])""")

_TOKEN_BEFORE_BOUNDARY = re.compile(r"""([A-Za-z0-9.]+)[.!?]["')\]]*\s*$""")

_ABBREVIATIONS = frozenset(
    "mr ms mrs dr jr sr st inc ltd co corp vs no vol approx dept est fig figs "
    "gov rep sen prof univ jan feb mar apr jun jul aug sep sept oct nov dec "
    "u.s u.k e.g i.e etc al cf ca div".split()
)


def _repair_entities(text: str) -> str:
    text = _NUMERIC_ENTITY_FRAGMENT.sub(r"&#\1;", text)
    text = _NAMED_ENTITY_FRAGMENT.sub(r"&\1;", text)
    return html.unescape(text)


def clean_text(text: str) -> str:
    """Normalise one document body into retrieval-ready plain text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ESCAPED_LITERALS.sub(r"\1", text)
    text = _STRAY_BACKSLASH.sub(" ", text)
    text = _repair_entities(text)
    text = _CONTROL_CHARS.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def _is_real_boundary(text: str, match: re.Match[str]) -> bool:
    token_match = _TOKEN_BEFORE_BOUNDARY.search(text[: match.end()])
    if token_match is None:
        return True
    token = token_match.group(1).lower().rstrip(".")
    if token in _ABBREVIATIONS:
        return False
    if len(token) == 1 and token.isalpha():
        return False  # an initial, as in "J. Smith"
    if token.isdigit():
        return False  # a decimal figure or a numbered list item
    return True


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if not _is_real_boundary(text, match):
            continue
        piece = text[start : match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_oversized(sentence: str, limit: int) -> list[str]:
    """Break a sentence longer than ``limit`` at word boundaries.

    Needed because some documents contain long spans with no usable sentence
    boundary at all. Press-release contact blocks are the worst case: every
    period sits inside an email address or follows an abbreviation ("Ms. Cara
    O'Brien ... cara.obrien@fticonsulting.com Tel: +852-..."), so the splitter
    correctly finds no boundary and the whole span arrives as one sentence.

    Measured before this guard, 7.1% of financial chunks exceeded their budget
    and the largest reached 13,066 characters. That is actively harmful rather
    than merely untidy: a 13k-character span compressed into a 384-dimension
    vector is semantic mush, and the cross-encoder silently truncates it at 512
    tokens, so the rerank score describes only its opening. Splitting on
    whitespace is a blunt instrument, but it is applied only to spans that have
    already defeated sentence segmentation.
    """
    words = sentence.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_len + addition > limit:
            pieces.append(" ".join(current))
            current, current_len = [], 0
            addition = len(word)
        current.append(word)
        current_len += addition

    if current:
        pieces.append(" ".join(current))
    # A single word longer than the limit is left intact; truncating it would
    # destroy a URL or an identifier without making the chunk meaningfully usable.
    return pieces or [sentence]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack sentences into chunks of at most ``chunk_size`` characters.

    ``chunk_overlap`` is a character budget met by repeating whole trailing
    sentences at the head of the next chunk, so no chunk begins mid-sentence.
    A span that exceeds ``chunk_size`` on its own is split at word boundaries by
    ``_split_oversized`` rather than being emitted as one outsized chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            sentences.extend(_split_oversized(sentence, chunk_size))
        else:
            sentences.append(sentence)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > chunk_size:
            chunks.append(" ".join(current))
            current, current_len = _carry_overlap(current, chunk_overlap)
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


def _carry_overlap(sentences: list[str], budget: int) -> tuple[list[str], int]:
    """Return the trailing sentences that fit in ``budget``, with their length."""
    if budget <= 0:
        return [], 0
    carried: list[str] = []
    length = 0
    for sentence in reversed(sentences):
        addition = len(sentence) + (1 if carried else 0)
        if length + addition > budget:
            break
        carried.insert(0, sentence)
        length += addition
    return carried, length
