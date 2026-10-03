"""Retrieve, then answer with citations, or say the documents don't contain the answer."""

import json
import re
import time
from dataclasses import dataclass, field

from documind.chunking.chunkers import Chunk
from documind.generation.llm import LLMClient
from documind.generation.prompts import build_messages
from documind.retrieval.search import Hit, Retriever

NOT_FOUND_MESSAGE = "I couldn't find this in the RBI documents I have."
CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    n: int
    chunk_id: str
    citation: str
    text: str
    url: str


@dataclass
class Answer:
    question: str
    answer: str
    found: bool
    sources: list[Source]  # only the sources the answer cites
    retrieved: list[Source]  # everything that was given to the model
    invalid_citations: list[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)


def to_source(n: int, chunk: Chunk) -> Source:
    return Source(n, chunk.chunk_id, chunk.citation(), chunk.text, chunk.source_url)


def parse_reply(text: str) -> dict:
    """Read the model's JSON, tolerating code fences or text around it."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    data = json.loads(match.group(0))
    return {
        "found": bool(data.get("found", True)),
        "answer": str(data.get("answer", "")).strip(),
        "citations": [int(c) for c in data.get("citations", []) if str(c).isdigit()],
    }


def check_citations(answer: str, listed: list[int], n_sources: int) -> tuple[list[int], list[int]]:
    """Citations used in the text plus those listed, split into valid and invalid numbers."""
    used = {int(n) for n in CITATION_RE.findall(answer)} | set(listed)
    valid = sorted(n for n in used if 1 <= n <= n_sources)
    invalid = sorted(n for n in used if not 1 <= n <= n_sources)
    return valid, invalid


class Answerer:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def answer(self, question: str, mode: str | None = None, k: int | None = None) -> Answer:
        started = time.perf_counter()
        hits: list[Hit] = self.retriever.search(question, mode, k)
        retrieval_seconds = time.perf_counter() - started
        chunks = [hit.chunk for hit in hits]
        retrieved = [to_source(n, c) for n, c in enumerate(chunks, 1)]

        completion = self.llm.chat(build_messages(question, chunks), json_mode=True)
        try:
            reply = parse_reply(completion.text)
        except (ValueError, json.JSONDecodeError):
            reply = {"found": False, "answer": "", "citations": []}

        valid, invalid = check_citations(reply["answer"], reply["citations"], len(chunks))
        found = reply["found"] and bool(reply["answer"]) and bool(valid)
        return Answer(
            question=question,
            answer=reply["answer"] if found else (reply["answer"] or NOT_FOUND_MESSAGE),
            found=found,
            sources=[retrieved[n - 1] for n in valid] if found else [],
            retrieved=retrieved,
            invalid_citations=invalid,
            timings={
                "retrieval_s": round(retrieval_seconds, 3),
                "llm_s": completion.seconds,
                "total_s": round(time.perf_counter() - started, 3),
            },
            tokens={"prompt": completion.prompt_tokens, "completion": completion.completion_tokens},
        )

    def close(self) -> None:
        self.retriever.close()
        self.llm.close()
