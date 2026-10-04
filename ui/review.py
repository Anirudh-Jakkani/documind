"""Review drafted test questions: keep, edit or reject each one.

    uv run streamlit run ui/review.py

Every click saves to eval/candidates.jsonl straight away, so you can stop and come back.
When done, "Export" writes the kept questions to eval/dataset.jsonl.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from documind.chunking.chunkers import short_title  # noqa: E402
from eval.testset import (  # noqa: E402
    CANDIDATES_PATH,
    DATASET_PATH,
    KINDS,
    key_span,
    load_records,
    normalise,
    save_records,
)

st.set_page_config(page_title="DocuMind · Test set review", layout="wide")

KIND_LABEL = {
    "single": "Single passage",
    "multi": "Needs two passages",
    "unanswerable": "Should NOT be answerable",
}
STATUS_ICON = {"pending": "⏳", "approved": "✅", "edited": "✏️", "rejected": "❌"}


@st.cache_data
def load_docs() -> dict:
    docs = {}
    for path in (ROOT / "data" / "processed").glob("rbi-*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        docs[doc["doc_id"]] = doc
    return docs


def source_paragraph(doc: dict, quote: str) -> dict | None:
    span = key_span(quote)
    for p in doc["paragraphs"]:
        if span in normalise(p["text"]):
            return p
    return None


records = load_records(CANDIDATES_PATH)
if not records:
    st.warning("No candidates yet. Run: `uv run python -m eval.generate`")
    st.stop()
docs = load_docs()

# --- Sidebar: progress, filters, export ---------------------------------------------------
with st.sidebar:
    st.header("Progress")
    done = sum(r.status != "pending" for r in records)
    st.progress(done / len(records), text=f"{done} of {len(records)} reviewed")
    for kind in KINDS:
        group = [r for r in records if r.kind == kind]
        if group:
            kept = sum(r.status in ("approved", "edited") for r in group)
            pending = sum(r.status == "pending" for r in group)
            st.caption(f"**{KIND_LABEL[kind]}**: {kept} kept · {pending} to review")

    st.header("Show")
    kinds = st.multiselect("Kind", KINDS, default=list(KINDS), format_func=KIND_LABEL.get)
    only_pending = st.toggle("Only questions still to review", value=True)

    st.header("Finish")
    kept_records = [r for r in records if r.status in ("approved", "edited")]
    if st.button(f"Export {len(kept_records)} kept questions", width="stretch"):
        save_records(kept_records, DATASET_PATH)
        st.success(f"Saved {len(kept_records)} questions to eval/{DATASET_PATH.name}")

visible = [
    i
    for i, r in enumerate(records)
    if r.kind in kinds and (r.status == "pending" or not only_pending)
]
if not visible:
    st.success("Nothing left to review with these filters. 🎉 Export from the sidebar.")
    st.stop()

if "pos" not in st.session_state:
    st.session_state.pos = 0
st.session_state.pos = min(st.session_state.pos, len(visible) - 1)
index = visible[st.session_state.pos]
record = records[index]

# --- Current record -----------------------------------------------------------------------
st.caption(
    f"{STATUS_ICON[record.status]} {record.id} · {KIND_LABEL[record.kind]} · "
    f"{record.qtype or '—'} · {record.origin} · {st.session_state.pos + 1} of {len(visible)}"
)

if "CHECK:" in record.note:
    st.warning(
        "DocuMind answered this question, so the documents may cover it after all. "
        "Reject it if the answer below is correct.\n\n"
        + record.note.split("CHECK:", 1)[1].split(" | WARN:")[0]
    )
if "WARN:" in record.note:
    st.warning("⚠️ Look closely: " + record.note.split("WARN:", 1)[1])
if record.note.startswith("reason:"):
    st.caption(record.note.split(" | ")[0])

left, right = st.columns([3, 2], gap="large")
with left:
    question = st.text_area("Question", record.question, height=90, key=f"q-{record.id}")
    if record.kind == "unanswerable":
        st.info("Expected behaviour: DocuMind should say the documents don't cover this.")
        answer = record.answer
    else:
        answer = st.text_area("Correct answer", record.answer, height=110, key=f"a-{record.id}")

    st.markdown("**Is this a good test question?**")
    st.caption(
        "Keep it if it's realistic, makes sense on its own, and the answer is correct and fully "
        "supported by the evidence on the right."
    )
    keep, save, reject, skip = st.columns(4)

    def advance():
        if not only_pending:
            st.session_state.pos += 1

    if keep.button("✅ Keep", width="stretch", type="primary"):
        record.status = "approved"
        save_records(records, CANDIDATES_PATH)
        advance()
        st.rerun()
    if save.button("✏️ Save edits", width="stretch"):
        record.question, record.answer = question.strip(), answer.strip()
        record.status = "edited"
        save_records(records, CANDIDATES_PATH)
        advance()
        st.rerun()
    if reject.button("❌ Reject", width="stretch"):
        record.status = "rejected"
        save_records(records, CANDIDATES_PATH)
        advance()
        st.rerun()
    if skip.button("Skip ⏭", width="stretch"):
        st.session_state.pos = (st.session_state.pos + 1) % len(visible)
        st.rerun()

with right:
    if not record.evidence:
        st.markdown("**Evidence**")
        st.caption("None: this question should not be answerable from the documents.")
    for n, ev in enumerate(record.evidence, 1):
        doc = docs.get(ev.doc_id, {})
        title = short_title(doc.get("title", ev.doc_id))
        st.markdown(f"**Evidence {n}** · {title}, para {ev.para or '—'}, p. {ev.page}")
        st.success(ev.quote)
        if doc:
            paragraph = source_paragraph(doc, ev.quote)
            with st.expander("Full paragraph in the document"):
                if paragraph:
                    st.caption(
                        " > ".join(x for x in (paragraph["chapter"], paragraph["heading"]) if x)
                    )
                    st.write(paragraph["text"])
                else:
                    st.caption("Paragraph not found.")
                st.link_button("Open on rbi.org.in", doc.get("source_url", ""))
