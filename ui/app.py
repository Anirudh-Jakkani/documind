"""DocuMind web app.

uv run streamlit run ui/app.py
"""

import json
import os
import re
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def secrets_to_environment() -> None:
    """On Streamlit Community Cloud, keys are added as app secrets. Copy top-level secrets
    into the environment, where DocuMind's settings read them (an existing value wins)."""
    try:
        secrets = dict(st.secrets)
    except Exception:  # no secrets file: running locally with .env
        return
    for key, value in secrets.items():
        if isinstance(value, str | int | float | bool):
            os.environ.setdefault(key.upper(), str(value))


secrets_to_environment()  # before anything reads the settings

from documind.config import get_settings  # noqa: E402
from documind.generation.llm import DailyLimitError, LLMError  # noqa: E402
from documind.limits import RateLimiter  # noqa: E402
from documind.observability import new_request_id, setup_logging  # noqa: E402
from documind.service import get_documind, load_documents  # noqa: E402

SETUP_HELP = """DocuMind can't reach its answer model: **GEMINI_API_KEY is not set.**

On Streamlit Community Cloud: **Manage app → ⋮ → Settings → Secrets**, add this line at the top
level (not under a `[section]`), save, then reboot the app:

```toml
GEMINI_API_KEY = "your-key-from-aistudio.google.com"
```

Running locally: put `GEMINI_API_KEY=...` in the `.env` file."""

RESULTS = ROOT / "eval" / "results"
EXAMPLES = [  # everyday wording; each answer ranks 1st or 2nd, so it holds on any machine
    "How much money can I send abroad in a year under LRS?",
    "If my fixed deposit matures on a holiday, how is interest paid for the extra days?",
    "Can the bank force me to use their mobile app just to get a debit card?",
    "What happens if my savings account falls below the minimum balance?",
    "What is the RBI's current repo rate?",
]

st.set_page_config(page_title="DocuMind", page_icon="📑", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .cite {background: rgba(92,160,255,.15); border-radius: 4px; padding: 0 4px;
           font-size: .85em; font-weight: 600;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the search index and models (first start only)…")
def load_service():
    setup_logging(get_settings().log_level)
    return get_documind()


def service():
    """The loaded DocuMind, or None (with a setup message) if it can't start."""
    if not get_settings().llm_configured:
        st.error(SETUP_HELP)
        return None
    try:
        return load_service()
    except LLMError as error:
        st.error(f"DocuMind couldn't start the answer model: {error}")
        return None


@st.cache_resource
def limiter() -> RateLimiter:
    """Shared by all sessions: each browser session gets RATE_LIMIT_PER_MINUTE questions."""
    return RateLimiter(get_settings().rate_limit_per_minute)


def highlight_citations(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r"<span class='cite'>[\1]</span>", text)


def render_answer(result: dict) -> None:
    if result["found"]:
        st.markdown(highlight_citations(result["answer"]), unsafe_allow_html=True)
        for source in result["sources"]:
            with st.expander(f"[{source['n']}] {source['citation']}"):
                st.write(source["text"])
                st.link_button("Open on rbi.org.in", source["url"])
    else:
        st.info(result["answer"])
        st.caption("DocuMind only answers from the RBI Directions it has. It won't guess.")
    t = result["timings"]
    st.caption(
        f"{t['total_s']:.1f} s · search {t['retrieval_s']:.1f} s · answer {t['llm_s']:.1f} s"
    )


def ask_tab() -> None:
    st.session_state.setdefault("history", [])
    if not st.session_state.history:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES, strict=True):
            if col.button(example, width="stretch"):
                st.session_state.pending = example
                st.rerun()

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_answer(item)

    question = st.chat_input("Ask about RBI rules for banks, cards, payments or forex…")
    question = question or st.session_state.pop("pending", None)
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    session = st.session_state.setdefault("session_id", new_request_id())
    wait = limiter().check(session)
    if wait:
        st.warning(f"That's a lot of questions in a minute. Please wait {wait:.0f} s.")
        return
    with st.chat_message("assistant"):
        with st.spinner("Searching the Directions…"):
            documind = service()
            if documind is None:
                return
            try:
                answer = documind.ask(question)
            except DailyLimitError:
                st.error("The free daily limit of the answer model is used up. Try again later.")
                return
            except LLMError as error:
                st.error(f"The answer model is unavailable right now ({error}).")
                return
        item = {
            "question": question,
            "answer": answer.answer,
            "found": answer.found,
            "sources": [s.__dict__ for s in answer.sources],
            "timings": answer.timings,
        }
        render_answer(item)
    st.session_state.history.append(item)


def load_summaries() -> dict:
    return {
        p.stem.removeprefix("summary_"): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(RESULTS.glob("summary_*.json"))
    }


def setup_label(row) -> str:
    """'hybrid / bge-small / minilm / yes' -> 'Hybrid + rewriting + MiniLM reranker'."""
    label = {"bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid"}[row["mode"]]
    if "base" in str(row["embedding"]):
        label += " (bge-base)"
    if row["rewrite"] == "yes":
        label += " + rewriting"
    if row["rerank"] != "none":
        label += {"minilm": " + MiniLM reranker", "bge": " + bge reranker"}[row["rerank"]]
    return label


def evaluation_tab() -> None:
    summaries = load_summaries()
    best = summaries.get("best") or summaries.get("baseline")
    if best:
        a, u = best["answerable"], best["unanswerable"]
        st.subheader("Answer quality")
        st.caption(
            f"{a['n'] + u['n']} reviewed test questions · answers by {best['answer_model']} · "
            f"graded by {best['judge_model']}"
        )
        c = st.columns(5)
        c[0].metric("Correct", f"{100 * a['correct']:.0f}%")
        c[1].metric("Faithful to sources", f"{100 * a['faithful_when_answered']:.0f}%")
        c[2].metric("Refuses when it should", f"{100 * u['correct_refusal']:.0f}%")
        c[3].metric("Median latency", f"{best['latency_s']['p50']} s")
        c[4].metric("False refusals", f"{100 * a['false_refusal']:.0f}%")

    if summaries:
        st.subheader("Configurations compared")
        rows = []
        for name, s in summaries.items():
            a = s["answerable"]
            rows.append(
                {
                    "run": name,
                    "questions": s.get("dataset", "original"),
                    "reranker": s.get("rerank", "none"),
                    "query rewriting": s.get("rewrite", "no"),
                    "correct": a["correct"],
                    "correct or partial": a["correct_or_partial"],
                    "faithful": a["faithful_when_answered"],
                    "recall@5": a["retrieval"].get("recall@5"),
                    "correct refusals": s["unanswerable"]["correct_refusal"],
                    "p50 latency (s)": s["latency_s"]["p50"],
                }
            )
        percent = st.column_config.NumberColumn(format="percent")
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                c: percent
                for c in ("correct", "correct or partial", "faithful", "correct refusals")
            },
        )

    path = RESULTS / "retrieval.csv"
    if path.exists():
        st.subheader("Retrieval experiments")
        st.caption(
            "Share of evidence found in the top 5 chunks (recall@5) for 102 answerable "
            "questions. 'Paraphrased' asks the same questions in everyday words."
        )
        df = pd.read_csv(path).fillna({"dataset": "original", "rerank": "none", "rewrite": "no"})
        df = df[df["strategy"] == "section"]
        df["setup"] = df.apply(setup_label, axis=1)
        order = (
            df[df["dataset"] == "paraphrased"].sort_values("recall@5", ascending=False)["setup"]
        ).tolist()
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                y=alt.Y("setup:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
                x=alt.X("recall@5:Q", scale=alt.Scale(domain=[0, 1]), title="recall@5"),
                yOffset="dataset:N",
                color=alt.Color(
                    "dataset:N",
                    title="Questions",
                    scale=alt.Scale(
                        domain=["original", "paraphrased"], range=["#5ca0ff", "#f5a623"]
                    ),
                ),
                tooltip=["setup", "dataset", "recall@5", "mrr", "ms_per_query"],
            )
            .properties(height=alt.Step(13))
        )
        st.altair_chart(chart, width="stretch")
        with st.expander("All retrieval results"):
            st.dataframe(df.drop(columns=["setup"]), hide_index=True)


def how_tab() -> None:
    st.subheader("How an answer is made")
    st.graphviz_chart(
        """digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontsize=11];
        q [label="Question"]; rw [label="Rewrite into\\nregulation wording\\n(LLM)"];
        s [label="Hybrid search\\nBM25 + embeddings\\n(top 30)"];
        rr [label="Rerank\\ncross-encoder\\n(top 5)"];
        a [label="Answer only from\\nthe 5 chunks,\\nwith [n] citations"];
        nf [label="'Not found'\\nif unsupported", fillcolor="#fff3e0"];
        q -> rw -> s -> rr -> a; a -> nf [style=dashed];
        }"""
    )
    st.markdown(
        """
- **Documents:** 42 RBI Directions (commercial banks, payments, foreign exchange, financial
  inclusion), parsed into paragraphs that keep their chapter, paragraph number and page.
- **Chunks** follow the documents' own sections, and each starts with a
  *document > chapter > heading* line, so search knows where a passage sits.
- **Search** combines keyword matching (BM25) with meaning-based search (bge embeddings),
  after an LLM rewrites everyday words into the regulation's terms ("fees" → "charges levied").
- **Answers** may only use the retrieved text. Every claim must cite a source; an answer
  without a valid citation is shown as "not found" instead.
- **Measured, not assumed:** every choice above was picked by experiments on a reviewed test
  set of 128 questions, including 26 the documents can't answer (see *Evaluation*).
"""
    )


def documents_tab() -> None:
    docs = pd.DataFrame(load_documents(get_settings()))  # works even without an API key
    st.caption(f"{len(docs)} RBI Directions. DocuMind answers only from these.")
    st.dataframe(
        docs[["topic", "title", "issued", "url"]],
        hide_index=True,
        width="stretch",
        column_config={"url": st.column_config.LinkColumn("RBI page", display_text="Open")},
    )


st.title("📑 DocuMind")
st.caption("Ask questions about RBI regulations. Every answer cites the paragraph it comes from.")
ask, evaluation, how, documents = st.tabs(["Ask", "Evaluation", "How it works", "Documents"])
with ask:
    ask_tab()
with evaluation:
    evaluation_tab()
with how:
    how_tab()
with documents:
    documents_tab()
st.caption("Educational project, not legal or financial advice. Always check the source.")
