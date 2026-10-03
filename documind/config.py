"""All settings in one place, read from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # Paths
    data_dir: Path = ROOT_DIR / "data"
    index_dir: Path = ROOT_DIR / "data" / "index"

    # LLM: any OpenAI-compatible API (OpenRouter, Groq, ...)
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1500  # reasoning models spend part of this on thinking
    llm_reasoning_effort: str = "low"  # sent only to models that support it (gpt-oss)

    # Embeddings and reranking (run locally)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Chunking
    chunk_strategy: str = Field(default="section", pattern="^(fixed|recursive|section)$")
    chunk_size_tokens: int = 400  # bge models read at most 512 tokens; leaves room for headers
    chunk_overlap_tokens: int = 60  # fixed strategy only
    section_min_tokens: int = 150  # section strategy: break at a new paragraph after this

    # Retrieval
    retrieval_mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")
    top_k_candidates: int = 30
    top_k_final: int = 5
    use_reranker: bool = True

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
