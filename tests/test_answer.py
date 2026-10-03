import json

from documind.chunking.chunkers import Chunk
from documind.generation.answer import (
    NOT_FOUND_MESSAGE,
    Answerer,
    check_citations,
    parse_reply,
)
from documind.generation.llm import Completion
from documind.generation.prompts import build_messages
from documind.retrieval.search import Hit


def chunk(n: int) -> Chunk:
    return Chunk(
        chunk_id=f"rbi-1:section:{n:04d}",
        doc_id="rbi-1",
        doc_title="Credit Cards",
        topic="Banking",
        strategy="section",
        text=f"Rule number {n}.",
        chapter="Chapter II",
        paras=[str(n)],
        page_start=n,
        page_end=n,
        source_url="https://rbi.org.in/x",
    )


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, question, mode=None, k=None, *, reranker=None, extra_queries=None):
        self.calls.append({"question": question, "extra_queries": extra_queries})
        return [Hit(chunk(1), 1.0), Hit(chunk(2), 0.5)]

    def close(self):
        pass


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = None

    def chat(self, messages, **kwargs):
        self.messages = messages
        return Completion(self.reply, 900, 60, 0.4, "fake")

    def close(self):
        pass


def ask(reply: dict | str):
    text = reply if isinstance(reply, str) else json.dumps(reply)
    llm = FakeLLM(text)
    return Answerer(FakeRetriever(), llm).answer("question?"), llm


def test_parse_reply_tolerates_code_fences():
    reply = parse_reply(
        '```json\n{"found": true, "answer": "Seven days [1].", "citations": [1]}\n```'
    )
    assert reply == {"found": True, "answer": "Seven days [1].", "citations": [1]}


def test_check_citations_splits_valid_and_invalid():
    assert check_citations("A [1]. B [2][7].", [3], n_sources=5) == ([1, 2, 3], [7])


def test_answer_keeps_only_cited_sources():
    result, _ = ask({"found": True, "answer": "Rule two applies [2].", "citations": [2]})
    assert result.found
    assert [s.n for s in result.sources] == [2]
    assert result.sources[0].citation == "Credit Cards, Chapter II, para 2, p. 2"
    assert len(result.retrieved) == 2


def test_not_found_reply():
    result, _ = ask({"found": False, "answer": "The sources do not cover this.", "citations": []})
    assert not result.found and result.sources == []
    assert result.answer == "The sources do not cover this."


def test_answer_without_valid_citation_is_treated_as_not_found():
    result, _ = ask({"found": True, "answer": "Banks must do X [9].", "citations": [9]})
    assert not result.found
    assert result.invalid_citations == [9]


def test_unparseable_reply_is_not_found():
    result, _ = ask("Sorry, I can't help with that.")
    assert not result.found and result.answer == NOT_FOUND_MESSAGE


def test_prompt_numbers_sources_with_citations():
    messages = build_messages("question?", [chunk(1), chunk(2)])
    user = messages[1]["content"]
    assert "[1] Credit Cards, Chapter II, para 1, p. 1\nRule number 1." in user
    assert "[2] Credit Cards" in user and "Question: question?" in user


class FakeRewriter:
    def rewrite(self, question):
        return ["formal version of " + question]


def test_rewritten_queries_are_searched_alongside_the_original():
    retriever = FakeRetriever()
    llm = FakeLLM(json.dumps({"found": True, "answer": "A [1].", "citations": [1]}))
    Answerer(retriever, llm, rewriter=FakeRewriter()).answer("fees on fraud?")
    assert retriever.calls == [
        {"question": "fees on fraud?", "extra_queries": ["formal version of fees on fraud?"]}
    ]
