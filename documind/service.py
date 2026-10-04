"""The DocuMind service: everything needed to answer a question, loaded once.

Used by the API (app/main.py) and the web app (ui/app.py). Loading the embedding model,
index and reranker takes several seconds, so create one DocuMind and reuse it.
"""

import csv
import logging
import time
from dataclasses import asdict, replace
from functools import lru_cache

from documind import __version__
from documind.config import Settings, get_settings
from documind.generation.answer import Answer, Answerer
from documind.generation.llm import LLMClient
from documind.limits import AnswerCache
from documind.observability import log_request, new_request_id
from documind.retrieval.reranker import Reranker
from documind.retrieval.rewrite import QueryRewriter
from documind.retrieval.search import Retriever

log = logging.getLogger("documind")


class DocuMind:
    def __init__(self, settings: Settings | None = None, answerer: Answerer | None = None):
        """`answerer` lets tests supply a stand-in; normally everything is loaded here."""
        started = time.perf_counter()
        self.settings = s = settings or get_settings()
        if answerer is None:
            retriever = Retriever(s.chunk_strategy, s)
            _ = retriever.embedder  # load the embedding model now, not on the first question
            answerer = Answerer(
                retriever,
                LLMClient(s),
                reranker=Reranker(s.reranker) if s.reranker != "none" else None,
                rewriter=QueryRewriter(LLMClient(s)) if s.query_rewriting else None,
            )
        self.answerer = answerer
        self.retriever, self.llm = answerer.retriever, answerer.llm
        self.reranker, self.rewriter = answerer.reranker, answerer.rewriter
        self.cache = AnswerCache(s.answer_cache_size, s.answer_cache_ttl_hours * 3600)
        self.documents = load_documents(s)
        self.load_seconds = round(time.perf_counter() - started, 1)
        log.info("DocuMind ready in %.1f s", self.load_seconds)

    def ask(self, question: str, request_id: str | None = None) -> Answer:
        """Answer a question, from the cache if it was asked recently. Every question is
        written to the request log, including failures."""
        question = question.strip()
        request_id = request_id or new_request_id()
        started = time.perf_counter()
        cached = self.cache.get(question)
        try:
            if cached is not None:
                answer = replace(
                    cached,
                    question=question,
                    timings={"total_s": round(time.perf_counter() - started, 3), "cached": True},
                    tokens={"prompt": 0, "completion": 0},
                )
            else:
                answer = self.answerer.answer(question)
                self.cache.put(question, answer)
        except Exception as error:
            log_request(
                self.settings.request_log,
                {
                    "request_id": request_id,
                    "question": question,
                    "error": repr(error)[:300],
                },
            )
            log.warning("request %s failed: %r", request_id, error)
            raise
        log_request(
            self.settings.request_log,
            {
                "request_id": request_id,
                "question": question,
                "found": answer.found,
                "cited": [s.citation for s in answer.sources],
                "retrieved": [s.chunk_id for s in answer.retrieved],
                "timings": answer.timings,
                "tokens": answer.tokens,
                "cached": cached is not None,
                "model": self.llm.model,
            },
        )
        log.info(
            "request %s %s in %.2f s%s",
            request_id,
            "answered" if answer.found else "not found",
            answer.timings.get("total_s", 0),
            " (cache)" if cached is not None else "",
        )
        return answer

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
            "answer_cache": f"{len(self.cache.items)} of {s.answer_cache_size}",
            "rate_limit_per_minute": s.rate_limit_per_minute,
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
