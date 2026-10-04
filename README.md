# DocuMind

[![CI](https://github.com/Anirudh-Jakkani/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/Anirudh-Jakkani/documind/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ani-documind.streamlit.app/)

**Live demo: [ani-documind.streamlit.app](https://ani-documind.streamlit.app/)** (free hosting:
after a quiet spell the app sleeps, and waking it takes about a minute)

Ask questions about a set of documents and get answers with citations (document, section, page).
If the answer isn't in the documents, DocuMind says so instead of guessing.

The retrieval and answer quality are **measured** on a reviewed test set of 128 questions, and every design
choice (chunking, embedding model, search method, reranker) is compared in a results table.

Demo documents: 42 RBI Directions (commercial banks, payments, foreign exchange, financial inclusion).

## Build plan

| Phase | What | Status |
|---|---|---|
| 0 | Project setup, config, health endpoint, tests | ✅ |
| 1 | Collect and parse documents | ✅ |
| 2 | Chunking and indexing (Qdrant + BM25) | ✅ |
| 3 | First working version with citations | ✅ |
| 4 | Test set (128 reviewed questions, incl. unanswerable) | ✅ |
| 5 | Evaluation harness and baseline scores | ✅ |
| 6 | Experiments and results table | ✅ |
| 7 | API and Streamlit interface | ✅ |
| 8 | Caching, logging, CI | ✅ |
| 9 | Deploy (Streamlit Community Cloud) | ✅ |

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

## Experiments (Phase 6): fixing retrieval

Phase 5 showed retrieval was the bottleneck, and that the test questions share vocabulary with
the passages (they were drafted from them). So every question was also **paraphrased into
everyday words** (`eval/dataset_paraphrased.jsonl`, same evidence) and each idea was tested on
both sets. Section chunks, top 5; recall@5 / MRR on the 102 answerable questions:

| Setup | Original questions | Paraphrased (everyday words) | Time / question (CPU) |
|---|---|---|---|
| BM25 | 0.926 / 0.872 | 0.613 / 0.462 | <0.1 s |
| Dense (bge-small) | 0.838 / 0.781 | 0.726 / 0.601 | 0.1 s |
| Dense (bge-base, 3× larger) | 0.858 / 0.750 | 0.716 / 0.564 | 0.2 s |
| Hybrid (baseline) | 0.922 / 0.848 | 0.691 / 0.553 | 0.1 s |
| Hybrid + MiniLM reranker | 0.922 / 0.835 | 0.730 / 0.621 | ~3 s |
| Hybrid + LLM query rewriting | 0.902 / 0.748 | 0.765 / 0.634 | +1 LLM call |
| **Hybrid + rewriting + MiniLM reranker** (shipped) | 0.922 / 0.835 | 0.765 / 0.637 | ~4 s |
| Hybrid + bge-reranker-base | **0.966 / 0.929** | **0.814 / 0.686** | ~23 s |

**Answer quality**, baseline vs the shipped setup (judge: Qwen3.8-27B):

| | Baseline | Rewriting + MiniLM reranker |
|---|---|---|
| Paraphrased: correct / correct-or-partial | 66.7% / 73.5% | **70.6% / 82.3%** |
| Paraphrased: false "not found" | 23.5% | **13.7%** |
| Original: correct | 86.3% | 86.3% |
| Original: multi-part fully correct | 62.5% | **75.0%** |
| Faithful · correct refusals (both sets) | 100% · 100% | 100% · 100% |
| Median latency | 1.4 s | 5.7 s |

**What I learned**
- **Keyword search collapses on everyday wording** (recall@5 0.93 → 0.61). Measuring only
  questions drafted from the documents would have hidden this.
- **A bigger embedding model didn't help** (bge-base ≈ bge-small, 3× slower to index). The
  problem was vocabulary, not model size.
- **LLM query rewriting** ("fees" → "charges levied") was the biggest single gain on everyday
  questions; the MiniLM reranker then recovers the ranking quality rewriting costs on the
  original questions.
- **bge-reranker-base is the most accurate** by a clear margin, but takes ~23 s per question on
  a laptop CPU. With a GPU it would be the default; on free CPU hosting the shipped setup is
  rewriting + MiniLM (~6 s end to end).
- Faithfulness and refusals stayed at 100% throughout: the gains come from finding the right
  text, not from letting the model guess.

Reproduce: `uv run python -m eval.run retrieval --strategies section --datasets original
paraphrased --modes hybrid --rewrite --rerank minilm` and
`uv run python -m eval.run answers --name best-paraphrased --dataset paraphrased --rewrite
--rerank minilm`.

## Running it as a service

- **Answer cache:** repeated questions (ignoring case, spacing and final punctuation) are
  answered instantly from an LRU cache (256 entries, 24 h), with no LLM call.
- **Rate limits:** 10 questions per minute per visitor (per IP for the API, per browser session
  for the web app), so one visitor can't use up the free daily LLM quota. The API answers
  `429` with `Retry-After`.
- **Request log:** one JSON line per question in `logs/requests.jsonl` (question, found,
  cited sources, retrieved chunk ids, timings, tokens, cached, request id; failures too),
  rotated at 5 MB. Every API answer carries an `X-Request-ID` header to match it to the log.
- **Index check:** search refuses to start if the vector index and the chunks don't match
  (this caught a real stale-index bug during Phase 6).
- **CI:** GitHub Actions runs ruff (lint and format) and the 66 tests on every push. Tests use
  stand-ins for the models, index and LLMs, so CI needs no data and no API keys.
- **CPU-only PyTorch** from PyTorch's CPU index keeps installs and images free of ~2.5 GB of
  unused CUDA libraries.

All limits are settings (`RATE_LIMIT_PER_MINUTE`, `ANSWER_CACHE_SIZE`, `REQUEST_LOG`, ...).

## Known limitations

- **Wording still matters, and borderline answers can flip.** "How long does a wallet company
  need to keep a record of all the transactions made through their wallets?" works on a laptop,
  but its answer paragraph ranks 4th of 5, and on the hosted app (Linux, different CPU) it falls
  out of the top 5. Rephrasing it ("How long must a wallet company keep records…") misses even
  locally, because the rewrite says "wallet issuers" instead of "PPI issuer". The example
  questions in the app were chosen because their answers rank 1st or 2nd.
- **Context the user leaves out.** "Can the bank charge me a fee for a transaction I reported
  as fraud?" fails: the rule is in the *credit card* directions ("charges levied on transactions
  disputed as fraud by the cardholder"), and without "credit card" in the question it never
  reaches the top 30. DocuMind says it can't find it rather than guessing.
- **Speed on CPU.** About 6 s per answer with rewriting and reranking (one extra LLM call plus
  a cross-encoder pass); the more accurate bge reranker takes ~23 s per question on a laptop.
- **Coverage.** 42 Directions for commercial banks, payments, forex and financial inclusion;
  rules for NBFCs, co-operative banks and small finance banks are out of scope by design.
- **Free tiers.** Answers use Gemini's free tier and grading uses Groq's; daily quotas limit how
  many evaluation runs fit in a day.

## Run locally

You need [uv](https://docs.astral.sh/uv/) and free API keys from Google AI Studio (answers) and
Groq (evaluation judge).

```bash
uv sync
cp .env.example .env                       # then add GEMINI_API_KEY and GROQ_API_KEY
uv run python -m documind.ingest.download --from-folder <folder with the RBI PDFs>
uv run python -m documind.ingest.parse     # PDFs -> structured paragraphs
uv run python -m documind.index --strategy section   # chunks, BM25 and vectors (~12 min on CPU)

uv run streamlit run ui/app.py             # web app: http://localhost:8501
uv run uvicorn app.main:app --port 8000    # API docs: http://localhost:8000/docs
uv run pytest                              # tests
```

`docs/download_links.html` lists the PDFs (RBI's server blocks scripted downloads, so they are
downloaded in a browser and imported from a folder).

## Project layout

```
app/          FastAPI service
documind/     core library: ingest, chunking, retrieval, generation, config
eval/         test set, metrics and experiment runner
ui/           Streamlit interface
data/         raw PDFs and processed text (not committed)
tests/        pytest suite
```
