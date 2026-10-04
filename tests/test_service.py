import json

import pytest

from documind.config import Settings
from documind.generation.answer import Answer, Source
from documind.service import DocuMind


class FakeAnswerer:
    retriever = llm = reranker = rewriter = None

    def __init__(self, fail: bool = False):
        self.calls, self.fail = 0, fail
        self.llm = type("LLM", (), {"model": "fake-model", "provider": "fake"})()

    def answer(self, question):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model down")
        source = Source(1, "rbi-1:section:0001", "Credit Cards, para 19, p. 9", "text", "url")
        return Answer(
            question,
            "Seven working days [1].",
            True,
            [source],
            [source],
            timings={"total_s": 1.5},
            tokens={"prompt": 900, "completion": 40},
        )


def make(tmp_path, fail=False):
    settings = Settings(_env_file=None, request_log=tmp_path / "requests.jsonl")
    answerer = FakeAnswerer(fail)
    return DocuMind(settings, answerer=answerer), answerer, tmp_path / "requests.jsonl"


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_repeated_question_is_answered_from_cache(tmp_path):
    documind, answerer, log_path = make(tmp_path)
    first = documind.ask("How fast must a card be closed?")
    second = documind.ask("how fast must a card be closed")
    assert answerer.calls == 1
    assert second.answer == first.answer and second.timings["cached"] is True
    assert second.tokens == {"prompt": 0, "completion": 0}
    log = read_log(log_path)
    assert [r["cached"] for r in log] == [False, True]
    assert log[0]["cited"] == ["Credit Cards, para 19, p. 9"] and log[0]["found"] is True


def test_failures_are_logged_and_raised(tmp_path):
    documind, _, log_path = make(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        documind.ask("Anything?", request_id="abc123")
    entry = read_log(log_path)[0]
    assert entry["request_id"] == "abc123" and "model down" in entry["error"]


def test_documents_are_listed(tmp_path):
    documind, _, _ = make(tmp_path)
    assert len(documind.documents) == 42
    assert {"doc_id", "title", "topic", "issued", "url"} <= set(documind.documents[0])
