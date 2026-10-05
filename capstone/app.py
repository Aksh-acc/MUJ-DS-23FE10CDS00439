"""Streamlit interface for the branched RAG market intelligence assistant.

The UI is built around showing the work, not just the answer. Every stage the
pipeline ran is on screen -- which branches the router chose and why, what query
each branch was actually sent, which retriever found each passage, and which
evidence the final answer cited. A RAG demo that shows only prose is
indistinguishable from a chatbot guessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from branched_rag.config import load_config  # noqa: E402
from branched_rag.pipeline import BranchedRAGPipeline, PipelineAnswer  # noqa: E402

st.set_page_config(
    page_title="Branched RAG Market Intelligence",
    layout="wide",
)

EXAMPLE_QUERIES = [
    "Should we launch a budget tablet? Consider demand, rivals and financial risk.",
    "What do customers complain about most in chargers and batteries?",
    "What is the revenue and profit outlook for the technology sector?",
    "Do product quality complaints line up with reported margin pressure?",
    "Which competitors announced new products, and what did analysts say?",
    "Give a full market assessment: customer feedback, competitor moves and financials.",
]

BRANCH_COLORS = {
    "reviews": "#2a9d8f",
    "news": "#e76f51",
    "financials": "#4b6cb7",
}


@st.cache_resource(show_spinner="Loading indexes and models...")
def get_pipeline() -> BranchedRAGPipeline:
    return BranchedRAGPipeline.load()


def index_missing_screen(error: Exception) -> None:
    st.title("Branched RAG Market Intelligence")
    st.error("The index has not been built yet.")
    st.code(
        "python scripts/build_corpus.py\npython scripts/build_index.py",
        language="bash",
    )
    with st.expander("Details"):
        st.write(str(error))


def render_sidebar(pipeline: BranchedRAGPipeline) -> dict:
    config = pipeline.config
    with st.sidebar:
        st.subheader("Generation backend")
        if pipeline.generation_available:
            st.success(pipeline.llm.description)
        else:
            st.warning("No backend available - retrieval only")
            st.caption(
                "Set GROQ_API_KEY in .env, start Ollama, or allow the local "
                "transformers model to download."
            )

        st.divider()
        st.subheader("Routing")
        strategy = st.selectbox(
            "Strategy",
            ["hybrid", "embedding", "lexical", "llm"],
            index=0,
            help=(
                "hybrid combines descriptor similarity with keyword overlap. "
                "See docs/results.md for how the four compare on the labelled set."
            ),
        )
        manual = st.multiselect(
            "Force branches (overrides the router)",
            options=config.branch_names,
            default=[],
            format_func=lambda name: config.branch(name).label,
            help="Use this to inspect a branch the router rejected.",
        )

        st.divider()
        st.subheader("Pipeline options")
        generate = st.toggle(
            "Generate answer",
            value=pipeline.generation_available,
            disabled=not pipeline.generation_available,
            help="Off runs retrieval only, which is fast and uses no tokens.",
        )
        expand_with_llm = st.toggle(
            "LLM query rewriting",
            value=False,
            disabled=not pipeline.generation_available,
            help=(
                "Off uses deterministic expansion from each branch's own "
                "vocabulary, which is reproducible and adds no latency."
            ),
        )

        st.divider()
        st.subheader("Corpus")
        stats = pipeline.store.stats()
        st.dataframe(
            stats[["label", "chunks", "share_of_corpus"]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"{pipeline.store.total_chunks:,} chunks. The uneven shares are why "
            "the branches are indexed separately."
        )

    return {
        "strategy": strategy,
        "branches": manual or None,
        "generate": generate,
        "expand_with_llm": expand_with_llm,
    }


def render_routing(answer: PipelineAnswer) -> None:
    plan = answer.plan
    st.markdown("#### Branch routing")

    decisions = plan.decisions
    figure = go.Figure(
        go.Bar(
            x=[d.score for d in decisions],
            y=[d.label for d in decisions],
            orientation="h",
            marker_color=[
                BRANCH_COLORS.get(d.branch, "#888") if d.selected else "#d7d7d7"
                for d in decisions
            ],
            text=[f"{d.score:.3f}" for d in decisions],
            textposition="outside",
            hovertext=[d.rationale for d in decisions],
            hoverinfo="text",
        )
    )
    figure.add_vline(
        x=plan.cutoff,
        line_dash="dot",
        line_color="#666",
        annotation_text=f"cutoff {plan.cutoff:.3f}",
        annotation_position="top",
    )
    figure.update_layout(
        height=190,
        margin=dict(l=0, r=40, t=10, b=0),
        xaxis_title="router score",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(figure, width="stretch")

    st.caption(f"Strategy: {plan.strategy}")
    if plan.fallback_applied:
        st.info(
            "No branch cleared the threshold, so the highest scoring branch was "
            "kept rather than returning no evidence."
        )

    for decision in plan.selected_decisions:
        st.markdown(f"- **{decision.label}** - {decision.rationale}")


def render_answer(answer: PipelineAnswer) -> None:
    fused = answer.answer
    if fused.failed and not fused.answer:
        st.error(f"Generation failed: {fused.error}")
        return
    if not fused.answer:
        st.info("Generation is off. The retrieved evidence is shown below.")
        return

    st.markdown(fused.answer)
    if fused.failed:
        st.warning(
            f"Fusion failed ({fused.error}); showing the branch summaries instead."
        )

    if fused.citations:
        lookup = answer.evidence_by_citation
        st.markdown("#### Cited sources")
        for number in fused.citations:
            item = lookup.get(number)
            if item is None:
                continue
            label = item.title or item.doc_id
            link = f" [source]({item.url})" if item.url else ""
            st.markdown(
                f"**[{number}]** {label} - _{item.branch}_, {item.source}{link}"
            )


def render_branch_detail(answer: PipelineAnswer) -> None:
    summaries = {s.branch: s for s in answer.answer.branch_summaries}
    if not answer.results:
        st.warning("No evidence retrieved.")
        return

    tabs = st.tabs([result.label for result in answer.results])
    for tab, result in zip(tabs, answer.results):
        with tab:
            summary = summaries.get(result.branch)
            if summary and summary.usable:
                st.markdown(summary.summary)
                st.divider()
            elif summary and summary.failed:
                st.warning(f"Branch synthesis failed: {summary.error}")

            left, right = st.columns(2)
            left.metric("Evidence used", len(result.evidence))
            right.metric("Candidates considered", result.candidates_considered)

            st.caption(
                f"Query sent to this branch: `{result.query.text}`"
                + (
                    f"  \nExpansion terms added: {', '.join(result.query.added_terms)}"
                    if result.query.added_terms
                    else ""
                )
                + f"  \nReranked: {'yes' if result.reranked else 'no'}"
            )

            for item in result.evidence:
                with st.expander(
                    f"[{item.citation}] {item.title or item.doc_id} "
                    f"({item.retrieval_path})"
                ):
                    st.write(item.text)
                    meta = {
                        "chunk": item.chunk_id,
                        "source": item.source,
                        "fused score": round(item.fused_score, 5),
                        "rerank score": (
                            round(item.rerank_score, 3)
                            if item.rerank_score is not None
                            else "n/a"
                        ),
                        "dense rank": item.dense_rank or "-",
                        "lexical rank": item.sparse_rank or "-",
                    }
                    if item.extra.get("stated_polarity"):
                        meta["dataset label"] = item.extra["stated_polarity"]
                    st.caption(
                        " | ".join(f"{key}: {value}" for key, value in meta.items())
                    )
                    if item.url:
                        st.markdown(f"[Open source]({item.url})")


def render_diagnostics(answer: PipelineAnswer) -> None:
    columns = st.columns(4)
    columns[0].metric("Branches used", len(answer.plan.selected))
    columns[1].metric("Evidence passages", len(answer.evidence))
    columns[2].metric(
        "Citation grounding",
        f"{answer.grounding_rate:.0%}" if answer.answer.citations else "n/a",
        help=(
            "Share of the answer's citation numbers that resolve to evidence "
            "actually retrieved. Below 100% means a source number was invented."
        ),
    )
    columns[3].metric("Total time", f"{answer.total_seconds:.2f}s")

    sentiment = answer.sentiment
    if sentiment.sampled:
        st.markdown("#### Review sentiment (retrieved sample)")
        bar = go.Figure(
            go.Bar(
                x=[sentiment.positive, sentiment.negative],
                y=["positive", "negative"],
                orientation="h",
                marker_color=["#2a9d8f", "#e76f51"],
                text=[sentiment.positive, sentiment.negative],
                textposition="outside",
            )
        )
        bar.update_layout(
            height=150, margin=dict(l=0, r=30, t=10, b=0), showlegend=False
        )
        st.plotly_chart(bar, width="stretch")
        st.caption(
            f"{sentiment.verdict} - {sentiment.positive_share:.0%} positive across "
            f"{sentiment.sampled} retrieved passages, mean classifier confidence "
            f"{sentiment.mean_confidence:.2f}. This describes the retrieved sample, "
            "not the market."
        )

    if answer.keyphrases:
        st.markdown("#### Query keyphrases")
        st.write(", ".join(f"`{phrase}`" for phrase in answer.keyphrases))

    st.markdown("#### Stage timings")
    timings = pd.DataFrame(
        [
            {"stage": stage, "seconds": round(value, 3)}
            for stage, value in answer.timings.items()
        ]
    )
    st.dataframe(timings, hide_index=True, width="stretch")


def main() -> None:
    try:
        pipeline = get_pipeline()
    except FileNotFoundError as error:
        index_missing_screen(error)
        return

    options = render_sidebar(pipeline)

    st.title("Branched RAG Market Intelligence")
    st.caption(
        "A question is routed to the sources that can answer it, each source is "
        "retrieved and summarised on its own terms, and the summaries are fused "
        "into one cited answer."
    )

    if "query" not in st.session_state:
        st.session_state.query = EXAMPLE_QUERIES[0]

    st.markdown("**Examples**")
    columns = st.columns(3)
    for position, example in enumerate(EXAMPLE_QUERIES):
        if columns[position % 3].button(
            example if len(example) < 52 else example[:49] + "...",
            key=f"example-{position}",
            width="stretch",
            help=example,
        ):
            st.session_state.query = example

    query = st.text_area("Question", key="query", height=86)
    run = st.button("Run", type="primary")

    if not run:
        return
    if not query.strip():
        st.warning("Enter a question first.")
        return

    with st.spinner("Routing, retrieving and synthesising..."):
        try:
            answer = pipeline.answer(
                query,
                strategy=options["strategy"],
                branches=options["branches"],
                expand_with_llm=options["expand_with_llm"],
                generate=options["generate"],
            )
        except Exception as error:
            st.error(f"Pipeline error: {error}")
            return

    answer_tab, branches_tab, routing_tab, diagnostics_tab = st.tabs(
        ["Answer", "Evidence by branch", "Routing", "Diagnostics"]
    )
    with answer_tab:
        render_answer(answer)
    with branches_tab:
        render_branch_detail(answer)
    with routing_tab:
        render_routing(answer)
    with diagnostics_tab:
        render_diagnostics(answer)


if __name__ == "__main__":
    main()
