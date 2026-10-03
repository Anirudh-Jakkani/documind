# DocuMind

Ask questions about a set of documents and get answers with citations (document, section, page).
If the answer isn't in the documents, DocuMind says so instead of guessing.

The retrieval and answer quality are **measured** on a reviewed test set of 128 questions, and every design
choice (chunking, embedding model, search method, reranker) is compared in a results table.

Demo documents: RBI Master Directions and FAQs.

> Work in progress. See the build plan below.

## Build plan

| Phase | What | Status |
|---|---|---|
| 0 | Project setup, config, health endpoint, tests | ✅ |
| 1 | Collect and parse documents | ✅ |
| 2 | Chunking and indexing (Qdrant + BM25) | ✅ |
| 3 | First working version with citations | ✅ |
| 4 | Test set (128 reviewed questions, incl. unanswerable) | ✅ |
| 5 | Evaluation harness and baseline scores | |
| 6 | Experiments and results table | |
| 7 | API and Streamlit interface | |
| 8 | Caching, logging, CI | |
| 9 | Deploy to Hugging Face Spaces | |

## Test set

`eval/dataset.jsonl` holds 128 questions over 42 RBI Directions:

| Kind | Count | What a correct system does |
|---|---|---|
| Single passage | 94 | Answers from one paragraph, with a citation |
| Multi-part | 8 | Combines two paragraphs (sometimes from two documents) |
| Unanswerable | 26 | Says the documents don't cover it (out of scope, false premise, too specific) |

**How it was made.** 180 candidates were drafted: single-passage and two-passage questions by
LLMs (gpt-oss-20b, Qwen3.8-27B) from sampled passages, plus hand-written hard and unanswerable
questions. Every candidate's evidence is a **verbatim quote** from the document; drafts whose
quote could not be found in the source were dropped automatically. The candidates were then
reviewed one by one by an AI assistant (Claude), which kept, edited or rejected each with a
recorded reason (see `eval/review_decisions.py` and the notes in `eval/candidates.jsonl`).
Most two-passage drafts were rejected because they glued unrelated facts together; five
natural multi-part questions were written by the reviewer instead.

**Why quotes, not chunk ids.** A retrieved chunk counts as relevant if it contains the middle
of the evidence quote, or, from the same document, at least half of it in one run (for when a
chunk boundary cuts the quote). The same test set therefore scores every chunking strategy.
Every kept quote was checked to be matchable in all three strategies' chunks.

## Run locally

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # then add your Gemini and Groq API keys
uv run pytest
uv run uvicorn app.main:app --reload   # http://localhost:8000/docs
```

## Project layout

```
app/          FastAPI service
documind/     core library: ingest, chunking, retrieval, generation, config
eval/         test set, metrics and experiment runner
ui/           Streamlit interface
data/         raw PDFs and processed text (not committed)
tests/        pytest suite
```
