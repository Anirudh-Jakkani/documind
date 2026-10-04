"""DocuMind API.

uv run uvicorn app.main:app --port 8000      # docs at http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from documind import __version__
from documind.config import get_settings
from documind.generation.llm import DailyLimitError, LLMError
from documind.limits import RateLimiter
from documind.observability import new_request_id, setup_logging
from documind.service import answer_to_dict, get_documind

limiter = RateLimiter(get_settings().rate_limit_per_minute)


def client_key(request: Request) -> str:
    """The visitor's IP. Behind a proxy (e.g. Hugging Face Spaces) it's the first address in
    X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(get_settings().log_level)
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
def ask(body: Question, request: Request, response: Response) -> dict:
    """Answer a question from the documents, citing document, chapter, paragraph and page.
    If the documents don't contain the answer, `found` is false. Limited to
    RATE_LIMIT_PER_MINUTE questions per minute per visitor."""
    wait = limiter.check(client_key(request))
    if wait:
        raise HTTPException(
            429,
            f"Too many questions. Please wait {wait:.0f} s.",
            headers={"Retry-After": str(int(wait) + 1)},
        )
    request_id = new_request_id()
    response.headers["X-Request-ID"] = request_id
    try:
        return answer_to_dict(get_documind().ask(body.question, request_id=request_id))
    except DailyLimitError as error:
        raise HTTPException(503, "The answer model's daily limit is used up.") from error
    except LLMError as error:
        raise HTTPException(502, f"The answer model is unavailable: {error}") from error
