from fastapi.testclient import TestClient

from app.main import app
from documind.config import Settings


def test_health_returns_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["retrieval_mode"] in {"dense", "bm25", "hybrid"}


def test_llm_not_configured_without_key():
    settings = Settings(_env_file=None, llm_api_key="", llm_model="")
    assert settings.llm_configured is False


def test_llm_configured_with_key_and_model():
    settings = Settings(_env_file=None, llm_api_key="test-key", llm_model="some/model")
    assert settings.llm_configured is True
