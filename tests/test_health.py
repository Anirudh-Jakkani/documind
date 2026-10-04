import pytest
from fastapi.testclient import TestClient

from app.main import app
from documind.config import Settings
from documind.generation.llm import LLMClient, LLMError, parse_model


def test_health_returns_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["retrieval_mode"] in {"dense", "bm25", "hybrid"}


def settings(**values) -> Settings:
    keys = {"gemini_api_key": "", "groq_api_key": "", "openrouter_api_key": ""}
    return Settings(_env_file=None, **{**keys, **values})


def test_llm_not_configured_without_key_for_its_provider():
    assert settings(llm_provider="gemini", groq_api_key="g").llm_configured is False


def test_llm_configured_with_key_for_its_provider():
    assert settings(llm_provider="gemini", gemini_api_key="k").llm_configured is True


def test_answer_and_judge_use_their_own_provider_and_key():
    s = settings(gemini_api_key="gem", groq_api_key="grq")
    answer, judge = LLMClient(s), LLMClient(s, role="judge")
    assert (answer.provider, answer.model) == ("gemini", s.llm_model)
    assert answer.http.headers["Authorization"] == "Bearer gem"
    assert (judge.provider, judge.model) == ("groq", s.judge_model)
    assert judge.http.headers["Authorization"] == "Bearer grq"
    assert str(judge.http.base_url).startswith("https://api.groq.com")


def test_model_override_with_provider_prefix():
    assert parse_model("groq:qwen/qwen3.8-27b", "gemini") == ("groq", "qwen/qwen3.8-27b")
    assert parse_model("openai/gpt-oss-20b", "groq") == ("groq", "openai/gpt-oss-20b")


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMClient(settings())


def test_index_dir_falls_back_to_the_committed_index(tmp_path):
    from documind.config import default_index_dir

    assert default_index_dir(tmp_path) == tmp_path / "deploy" / "index"  # fresh clone
    (tmp_path / "data" / "index" / "qdrant").mkdir(parents=True)
    assert default_index_dir(tmp_path) == tmp_path / "data" / "index"  # built locally
