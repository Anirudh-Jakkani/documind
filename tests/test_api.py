import pytest
from fastapi.testclient import TestClient

import app.main as api
from documind.generation.answer import Answer, Source
from documind.generation.llm import DailyLimitError
from documind.limits import RateLimiter


class FakeDocuMind:
    documents = [
        {
            "doc_id": "rbi-1",
            "title": "Credit Cards",
            "topic": "Banking",
            "issued": "2025",
            "url": "https://rbi.org.in/x",
        }
    ]

    def __init__(self, error: Exception | None = None):
        self.error = error

    def ask(self, question, request_id=None):
        if self.error:
            raise self.error
        source = Source(
            1,
            "rbi-1:section:0001",
            "Credit Cards, para 19, p. 9",
            "Closure within seven working days.",
            "https://rbi.org.in/x",
        )
        return Answer(
            question,
            "Within seven working days [1].",
            True,
            [source],
            [source],
            timings={"total_s": 1.0},
            tokens={"prompt": 10, "completion": 5},
        )

    def info(self):
        return {"documents": 1}

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    def use(fake, limit=100):
        monkeypatch.setattr(api, "get_documind", lambda: fake)
        monkeypatch.setattr(api, "limiter", RateLimiter(limit))
        return TestClient(api.app)

    return use


def test_ask_returns_answer_with_cited_sources(client):
    response = client(FakeDocuMind()).post("/ask", json={"question": "Card closure time?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found"] is True
    assert body["sources"][0]["citation"] == "Credit Cards, para 19, p. 9"
    assert "retrieved" not in body


def test_ask_validates_question_length(client):
    assert client(FakeDocuMind()).post("/ask", json={"question": "a"}).status_code == 422


def test_daily_limit_becomes_503(client):
    fake = FakeDocuMind(DailyLimitError("quota"))
    assert client(fake).post("/ask", json={"question": "Card closure time?"}).status_code == 503


def test_documents_and_info(client):
    c = client(FakeDocuMind())
    assert c.get("/documents").json()[0]["doc_id"] == "rbi-1"
    assert c.get("/info").json() == {"documents": 1}


def test_rate_limit_returns_429_with_retry_after(client):
    c = client(FakeDocuMind(), limit=1)
    first = c.post("/ask", json={"question": "Card closure time?"})
    assert first.status_code == 200 and first.headers["X-Request-ID"]
    second = c.post("/ask", json={"question": "Card closure time?"})
    assert second.status_code == 429 and int(second.headers["Retry-After"]) > 0
