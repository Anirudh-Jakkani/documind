"""The DocuMind service: everything needed to answer a question, loaded once.

Used by the API (app/main.py) and the web app (ui/app.py). Loading the embedding model,
index and reranker takes several seconds, so create one DocuMind and reuse it.
"""

import csv
import time
from dataclasses import asdict
from functools import lru_cache

from documind import __version__
from documind.config import Settings, get_settings
from documind.generation.answer import Answer, Answerer
from documind.generation.llm import LLMClient
from documind.retrieval.reranker import Reranker
from documind.retrieval.rewrite import QueryRewriter
from documind.retrieval.search import Retriever


class DocuMind:
    def __init__(self, settings: Settings | None = None):
        started = time.perf_counter()
        self.settings = s = settings or get_settings()
        self.retriever = Retriever(s.chunk_strategy, s)
        _ = self.retriever.embedder  # load the embedding model now, not on the first question
        self.llm = LLMClient(s)
        self.reranker = Reranker(s.reranker) if s.reranker != "none" else None
        self.rewriter = QueryRewriter(LLMClient(s)) if s.query_rewriting else None
        self.answerer = Answerer(
            self.retriever, self.llm, reranker=self.reranker, rewriter=self.rewriter
        )
        self.documents = load_documents(s)
        self.load_seconds = round(time.perf_counter() - started, 1)

    def ask(self, question: str) -> Answer:
        return self.answerer.answer(question.strip())

    def info(self) -> dict:
        s = self.settings
        return {
            "version": __version__,
            "documents": len(self.documents),
            "chunks": len(self.retriever.chunks),
            "chunking": s.chunk_strategy,
            "search": s.retrieval_mode,
            "embedding_model": s.embedding_model,
            "reranker": self.reranker.model_name if self.reranker else "none",
            "query_rewriting": s.query_rewriting,
            "answer_model": f"{self.llm.provider}:{self.llm.model}",
            "top_k": s.top_k_final,
            "load_seconds": self.load_seconds,
        }

    def close(self) -> None:
        self.answerer.close()


def load_documents(settings: Settings) -> list[dict]:
    with (settings.data_dir / "documents.csv").open(encoding="utf-8", newline="") as f:
        return [
            {
                "doc_id": row["doc_id"],
                "title": row["title"],
                "topic": row["topic"],
                "issued": row["issued"],
                "url": row["rbi_page"],
            }
            for row in csv.DictReader(f)
        ]


def answer_to_dict(answer: Answer) -> dict:
    data = asdict(answer)
    data.pop("retrieved")  # the cited sources are what a user needs
    return data


@lru_cache
def get_documind() -> DocuMind:
    return DocuMind()
