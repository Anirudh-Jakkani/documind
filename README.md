# DocuMind

Ask questions about a set of documents and get answers with citations (document, section, page).
If the answer isn't in the documents, DocuMind says so instead of guessing.

The retrieval and answer quality are **measured** on a hand-checked test set, and every design
choice (chunking, embedding model, search method, reranker) is compared in a results table.

Demo documents: RBI Master Directions and FAQs.

> Work in progress. See the build plan below.

## Build plan

| Phase | What | Status |
|---|---|---|
| 0 | Project setup, config, health endpoint, tests | ✅ |
| 1 | Collect and parse documents | ✅ |
| 2 | Chunking and indexing (Qdrant + BM25) | ✅ |
| 3 | First working version with citations | |
| 4 | Test set (~150 questions, incl. unanswerable) | |
| 5 | Evaluation harness and baseline scores | |
| 6 | Experiments and results table | |
| 7 | API and Streamlit interface | |
| 8 | Caching, logging, CI | |
| 9 | Deploy to Hugging Face Spaces | |

## Run locally

You need [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # then add your LLM key and model
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
