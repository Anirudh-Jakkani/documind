"""Minimal client for any OpenAI-compatible chat API (Groq, OpenRouter, ...).

Retries rate limits and server errors, honouring the provider's Retry-After header, and
reports token usage and latency for the cost and speed numbers in the evaluation.
"""

import time
from dataclasses import dataclass

import httpx

from documind.config import Settings, get_settings


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    model: str


class LLMClient:
    def __init__(
        self, settings: Settings | None = None, model: str | None = None, max_retries: int = 6
    ):
        self.settings = settings or get_settings()
        if not self.settings.llm_configured:
            raise LLMError("LLM not configured: set LLM_API_KEY and LLM_MODEL in .env")
        self.model = model or self.settings.llm_model
        self.max_retries = max_retries
        self.http = httpx.Client(
            base_url=self.settings.llm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
            timeout=120,
        )

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model and self.settings.llm_reasoning_effort:
            body["reasoning_effort"] = self.settings.llm_reasoning_effort

        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                if attempt == self.max_retries:
                    raise LLMError(f"network error: {error}") from error
                time.sleep(2**attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(f"{response.status_code} after {attempt + 1} attempts")
                time.sleep(retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise LLMError(f"{response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage") or {}
            return Completion(
                text=data["choices"][0]["message"]["content"] or "",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                seconds=round(time.perf_counter() - started, 2),
                model=data.get("model", self.model),
            )
        raise LLMError("unreachable")

    def close(self) -> None:
        self.http.close()


def retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    try:
        return min(float(header), 60.0) if header else min(2**attempt, 30)
    except ValueError:
        return min(2**attempt, 30)
