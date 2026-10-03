"""DocuMind API."""

from fastapi import FastAPI

from documind import __version__
from documind.config import get_settings

app = FastAPI(title="DocuMind API", version=__version__)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_configured": settings.llm_configured,
        "retrieval_mode": settings.retrieval_mode,
    }
