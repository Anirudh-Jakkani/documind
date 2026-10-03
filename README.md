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
| 5 | Evaluation harness and baseline scores | ✅ |
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

## Results (baseline)

Section chunks, hybrid search (BM25 + bge-small, reciprocal rank fusion), top 5 chunks.
Answers by `gemini-3.5-flash-lite`; graded by `qwen3.8-27b` (a different model family).

| Metric | Score |
|---|---|
| **Correct** (answerable questions, n=102) | **86.3%** (92.2% correct or partly correct) |
| **Faithful** (every claim supported by the cited sources, when answered) | **100%** |
| **Cites the evidence paragraph** (when answered) | 95.8% |
| **Correctly refuses** unanswerable questions (n=26) | **100%** |
| False refusals (said "not found" when the answer exists) | 6.9% |
| Multi-part questions fully correct (n=8) | 62.5% |
| Invalid citations | 0% |
| Latency, median / 95th percentile | 1.4 s / 2.2 s |
| Tokens per question (prompt + answer) | ~1,620 + 76 |

**The judge was audited:** a second reviewer re-graded 30 random answers and agreed on 30/30
faithfulness and 29/30 correctness verdicts ([eval/results/judge_audit.md](eval/results/judge_audit.md)).

**Where it fails.** Of the 13 answerable questions not fully correct, 7 had **recall@5 = 0**:
the right paragraph never reached the model, which then (correctly) refused or answered from
what it had. Example: asked for the shareholding that makes someone a "related person" (5%),
search returned a different related-party clause (10%), and the model faithfully answered 10%:
faithful, but wrong. **Retrieval, not generation, is the bottleneck.**

### Retrieval: every chunking strategy × search mode (102 answerable questions)

| Strategy | Search | recall@1 | recall@5 | MRR | ms / query |
|---|---|---|---|---|---|
| section | BM25 | **0.765** | **0.927** | **0.872** | 13 |
| recursive | hybrid | 0.681 | 0.927 | 0.801 | 125 |
| section | hybrid | 0.755 | 0.922 | 0.848 | 97 |
| fixed | BM25 | 0.627 | 0.922 | 0.768 | 24 |
| fixed | hybrid | 0.672 | 0.912 | 0.782 | 123 |
| recursive | BM25 | 0.627 | 0.907 | 0.770 | 19 |
| recursive | dense | 0.608 | 0.863 | 0.737 | 103 |
| fixed | dense | 0.608 | 0.858 | 0.717 | 111 |
| section | dense | 0.711 | 0.838 | 0.782 | 106 |

- **Section chunking puts the right paragraph first 75% of the time** (recall@1), versus
  61–68% for fixed and recursive chunks.
- **Keyword search beats dense search** on regulations full of exact terms ("FCNR(B)",
  "₹25 lakh", "seven working days"); hybrid keeps most of both.
- **Caveat:** test questions were drafted from the passages, so they share vocabulary with them,
  which favours BM25 and flatters recall. Real users phrase things differently (e.g. "fees"
  vs "charges levied"). Phase 6 tests paraphrased questions.

Reproduce: `uv run python -m eval.run retrieval` (free) and
`uv run python -m eval.run answers --name baseline`.

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
