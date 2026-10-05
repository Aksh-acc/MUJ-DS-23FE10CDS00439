"""Prompts for branch synthesis and cross-branch fusion.

Prompt specialisation is the second half of the argument for branching. Having
split retrieval by source, a single generic "answer using the context" prompt
throws the split away: the model gets one instruction for three kinds of
evidence and treats a first-person complaint, a news lede and an audited revenue
figure as interchangeable sentences.

Each branch therefore has its own analyst instruction describing what that
source can and cannot support:

* reviews can establish what users report, never market size
* news can establish that an event was reported and when, never its magnitude
* financials can establish reported figures, never why customers churn

Those limits are stated in the prompts because the most common RAG failure here
is not fabrication but overreach -- answering a demand question from three
reviews. The fusion prompt then gets the branch summaries rather than the raw
chunks, and is told to surface disagreement instead of averaging it, since
reviews and financials pointing opposite ways is a finding rather than noise.

Citations are plain bracketed integers assigned at retrieval time, which makes
grounding mechanically checkable: ``evaluation/metrics.py`` parses them back out
and verifies every one resolves to evidence actually supplied.
"""

from __future__ import annotations

from ..retrieval.retriever import BranchResult

_CITATION_RULES = """Citation rules:
- Support every factual claim with a bracketed source number, for example [3].
- Use plain ASCII square brackets only. Never write a citation as [3|L1-L2] or
  with any other bracket character or line reference.
- Cite only numbers that appear in the evidence below. Never invent one.
- If the evidence does not answer part of the question, say so plainly.
- Do not use outside knowledge. The evidence is the only permitted source."""

BRANCH_INSTRUCTIONS = {
    "reviews": """You are a consumer insight analyst reading customer reviews.

From this evidence you may report: what users praise and complain about, which
failure modes recur, how they describe value for money, and the balance of
opinion in the sample.

You may not report: market size, sales volumes, revenue, or what any company's
financial position is. Reviews cannot establish those.

Report recurring themes rather than retelling individual reviews, and say how
many of the cited reviews support each theme.""",
    "news": """You are a market analyst reading business and technology reporting.

From this evidence you may report: what was announced or reported, by whom and
when, competitor actions, partnerships, regulatory developments, and the
direction of a stated trend.

You may not report: audited financial figures unless the article states them, nor
customer sentiment. News reports claims; attribute them rather than asserting
them.

Prefer the specific over the general: name the companies and the events.""",
    "financials": """You are a financial analyst reading investor-facing reporting.

From this evidence you may report: revenue, profit, margin and guidance figures
as stated, along with analyst views, stakes and capital movements. Quote figures
exactly, with their period and currency.

You may not report: why customers behave as they do, or product quality. Those
are not in financial reporting.

Distinguish a reported result from a forecast, and note when a figure is a
company's own guidance rather than an outcome.""",
}

_DEFAULT_BRANCH_INSTRUCTION = """You are an analyst summarising the evidence below.
Report only what the evidence supports and attribute each claim to its source."""

BRANCH_SYSTEM = (
    "You are a precise research analyst. You ground every claim in the supplied "
    "evidence and you would rather report that evidence is missing than guess."
)

FUSION_SYSTEM = (
    "You are a senior market intelligence analyst writing for a decision maker. "
    "You combine findings from several sources, you are explicit about which "
    "source supports what, and you never smooth over a contradiction."
)


def format_evidence(result: BranchResult, max_chars: int) -> str:
    """Render one branch's evidence as a numbered block for the prompt."""
    lines = []
    for item in result.evidence:
        text = item.text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        header = f"[{item.citation}]"
        if item.title:
            header += f" {item.title}"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def branch_prompt(
    query: str, result: BranchResult, max_evidence_chars: int
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one branch's synthesis."""
    instruction = BRANCH_INSTRUCTIONS.get(result.branch, _DEFAULT_BRANCH_INSTRUCTION)
    evidence = format_evidence(result, max_evidence_chars)
    user = f"""{instruction}

{_CITATION_RULES}

Question: {query}

Evidence from {result.label}:

{evidence}

Write 3-5 sentences answering the question from this evidence alone."""
    return BRANCH_SYSTEM, user


def fusion_prompt(
    query: str,
    branch_summaries: list[tuple[str, str]],
    sentiment_note: str | None = None,
) -> tuple[str, str]:
    """Build the (system, user) prompt pair that merges branch summaries."""
    blocks = "\n\n".join(
        f"### {label}\n{summary}" for label, summary in branch_summaries
    )
    extra = f"\n\nQuantitative note:\n{sentiment_note}" if sentiment_note else ""

    user = f"""Question: {query}

You have been given one analyst summary per source. Each already carries source
numbers in brackets.

{blocks}{extra}

Write the final answer as follows:

1. A direct answer to the question, two to four sentences.
2. A short bullet per source heading, giving what that source establishes.
3. A line beginning "Agreement:" stating where the sources agree, and where they
   conflict or only one source speaks, say that explicitly.
4. A line beginning "Confidence:" with high, medium or low, and the reason, which
   should reference how many sources and how directly they address the question.

Carry the bracketed source numbers through unchanged, as plain ASCII [n]. Add no
new numbers. Do not introduce any fact that is not in the summaries above."""
    return FUSION_SYSTEM, user
