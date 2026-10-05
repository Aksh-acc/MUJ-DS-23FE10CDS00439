"""Branch synthesis and cross-branch fusion.

The generation half of the branch-then-fuse design. Each selected branch is
summarised independently from its own evidence under its own instruction, then a
single fusion call merges those summaries.

Fusion consumes summaries rather than raw chunks for two reasons. Raw chunks from
three branches overflow a useful context window and push the model toward
whichever source is longest, which is the same volume bias that separate indexes
were built to avoid. And a summary that has already been constrained to one
source's competence is easier to attribute: the fusion step can say "reviews
support X, financials do not" because the inputs are separated by construction.

Branch summaries are generated concurrently. They are independent by design, and
the latency of the branch stage is otherwise the sum of its calls rather than
the slowest one.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import Config
from ..retrieval.retriever import BranchResult
from .llm import LLMProvider, LLMUnavailable
from .prompts import branch_prompt, fusion_prompt

logger = logging.getLogger(__name__)


@dataclass
class BranchSummary:
    branch: str
    label: str
    summary: str
    citations: list[int] = field(default_factory=list)
    failed: bool = False
    error: str = ""

    @property
    def usable(self) -> bool:
        return not self.failed and bool(self.summary.strip())


@dataclass
class FusedAnswer:
    answer: str
    branch_summaries: list[BranchSummary]
    citations: list[int] = field(default_factory=list)
    failed: bool = False
    error: str = ""


class Synthesizer:
    def __init__(self, config: Config, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm

    @property
    def available(self) -> bool:
        return self.llm.available

    def summarise_branches(self, query: str, results: list[BranchResult]) -> list[BranchSummary]:
        populated = [result for result in results if not result.is_empty]
        if not populated:
            return []
        if not self.llm.available:
            return [
                BranchSummary(
                    branch=result.branch,
                    label=result.label,
                    summary="",
                    failed=True,
                    error="no generation backend available",
                )
                for result in populated
            ]

        with ThreadPoolExecutor(max_workers=len(populated)) as pool:
            return list(pool.map(lambda r: self._summarise_one(query, r), populated))

    def _summarise_one(self, query: str, result: BranchResult) -> BranchSummary:
        system, user = branch_prompt(
            query, result, self.config.generation.max_evidence_chars
        )
        try:
            text = self.llm.complete(
                user,
                system=system,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_tokens,
            )
        except Exception as error:
            logger.warning("branch %s synthesis failed: %s", result.branch, error)
            return BranchSummary(
                branch=result.branch,
                label=result.label,
                summary="",
                failed=True,
                error=str(error),
            )

        text = normalize_citation_markers(text)
        valid = {item.citation for item in result.evidence}
        return BranchSummary(
            branch=result.branch,
            label=result.label,
            summary=text,
            citations=sorted(extract_citations(text) & valid),
        )

    def fuse(
        self,
        query: str,
        summaries: list[BranchSummary],
        sentiment_note: str | None = None,
    ) -> FusedAnswer:
        usable = [summary for summary in summaries if summary.usable]
        if not usable:
            reason = next(
                (s.error for s in summaries if s.error), "no branch produced a summary"
            )
            return FusedAnswer(
                answer="",
                branch_summaries=summaries,
                failed=True,
                error=reason,
            )

        # One usable branch needs no fusion call. Paying for a second round trip
        # to restate a single summary buys nothing.
        if len(usable) == 1:
            only = usable[0]
            return FusedAnswer(
                answer=only.summary,
                branch_summaries=summaries,
                citations=only.citations,
            )

        system, user = fusion_prompt(
            query,
            [(summary.label, summary.summary) for summary in usable],
            sentiment_note=sentiment_note,
        )
        try:
            text = self.llm.complete(
                user,
                system=system,
                temperature=self.config.generation.temperature,
                max_tokens=self.config.generation.max_tokens,
            )
        except Exception as error:
            logger.warning("fusion failed: %s", error)
            # Falling back to the concatenated branch summaries keeps a usable
            # answer on screen when only the final call failed.
            joined = "\n\n".join(
                f"**{summary.label}**\n{summary.summary}" for summary in usable
            )
            return FusedAnswer(
                answer=joined,
                branch_summaries=summaries,
                citations=sorted({c for s in usable for c in s.citations}),
                failed=True,
                error=str(error),
            )

        text = normalize_citation_markers(text)
        allowed = {c for summary in usable for c in summary.citations}
        return FusedAnswer(
            answer=text,
            branch_summaries=summaries,
            citations=sorted(extract_citations(text) & allowed),
        )


_CITATION_GROUP = re.compile(r"\[([0-9,\s]+)\]")

# Models trained with a browsing tool fall back to that tool's citation syntax
# under pressure, and gpt-oss emits forms like 【5†L0-L2】 instead of [5] on some
# runs of the same prompt. Observed non-deterministically on identical input, so
# the prompt alone cannot be relied on: markers are normalised to ASCII before
# anything tries to parse or display them. Without this the citation numbers
# parse as empty and the grounding metric silently reads zero.
_EXOTIC_MARKERS = (
    re.compile(r"【\s*(\d+)\s*(?:†[^】]*)?】"),          # 【5†L0-L2】 and 【5】
    re.compile(r"\[\s*(\d+)\s*†[^\]]*\]"),              # [5†L0-L2]
    re.compile(r"【\s*(\d+)\s*[^】]*】"),   # any other full-width form
)


def normalize_citation_markers(text: str) -> str:
    """Rewrite non-ASCII citation markers as plain ``[n]``."""
    if not text:
        return ""
    for pattern in _EXOTIC_MARKERS:
        text = pattern.sub(r"[\1]", text)
    return text


def extract_citations(text: str) -> set[int]:
    """Pull bracketed citation numbers out of generated text.

    Handles both ``[3]`` and grouped forms such as ``[3, 7]`` or ``[3][7]``,
    which models produce interchangeably however the prompt asks.
    """
    found: set[int] = set()
    for group in _CITATION_GROUP.findall(normalize_citation_markers(text)):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))
    return found
