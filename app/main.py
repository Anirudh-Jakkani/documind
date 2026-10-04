"""DocuMind API.

uv run uvicorn app.main:app --port 8000      # docs at http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from documind import __version__
from documind.config import get_settings
from documind.generation.llm import DailyLimitError, LLMError
from documind.service import answer_to_dict, get_documind


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_documind()  # load models and index at startup, not on the first request
    yield
    get_documind().close()


app = FastAPI(
    title="DocuMind API",
    version=__version__,
    description="Questions about RBI Directions, answered with citations.",
    lifespan=lifespan,
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["What is the LRS limit?"])


class Source(BaseModel):
    n: int
    chunk_id: str
    citation: str
    text: str
    url: str


class AnswerOut(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[Source]
    invalid_citations: list[int]
    timings: dict
    tokens: dict


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_configured": settings.llm_configured,
        "retrieval_mode": settings.retrieval_mode,
    }


@app.get("/info")
def info() -> dict:
    """How this instance is configured."""
    return get_documind().info()


@app.get("/documents")
def documents() -> list[dict]:
    """The RBI Directions DocuMind can answer from."""
    return get_documind().documents


@app.post("/ask", response_model=AnswerOut)
def ask(body: Question) -> dict:
    """Answer a question from the documents, citing document, chapter, paragraph and page.
    If the documents don't contain the answer, `found` is false."""
    try:
        return answer_to_dict(get_documind().ask(body.question))
    except DailyLimitError as error:
        raise HTTPException(503, "The answer model's daily limit is used up.") from error
    except LLMError as error:
        raise HTTPException(502, f"The answer model is unavailable: {error}") from error
