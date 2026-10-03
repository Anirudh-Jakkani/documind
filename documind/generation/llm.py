"""Minimal client for OpenAI-compatible chat APIs (Gemini, Groq, OpenRouter).

Retries rate limits and server errors, honouring the provider's Retry-After header, switches
to a fallback model while the main one is overloaded, and reports token usage and latency
for the cost and speed numbers in the evaluation.
"""

import time
from dataclasses import dataclass

import httpx

from documind.config import PROVIDER_URLS, Settings, get_settings

OVERLOADED_TRIES_BEFORE_FALLBACK = 3


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    model: str


def parse_model(spec: str, default_provider: str) -> tuple[str, str]:
    """'groq:qwen/qwen3.8-27b' -> ('groq', 'qwen/qwen3.8-27b'); a bare model name uses the
    default provider."""
    provider, sep, model = spec.partition(":")
    return (provider, model) if sep and provider in PROVIDER_URLS else (default_provider, spec)


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        model: str | None = None,
        *,
        role: str = "answer",
        max_retries: int = 10,
    ):
        """role="answer" uses LLM_PROVIDER / LLM_MODEL, role="judge" uses JUDGE_*.
        `model` overrides the model, optionally with a provider prefix ("groq:...")."""
        self.settings = s = settings or get_settings()
        provider = s.llm_provider if role == "answer" else s.judge_provider
        default_model = s.llm_model if role == "answer" else s.judge_model
        self.provider, self.model = parse_model(model or default_model, provider)
        self.fallback_model = s.llm_fallback_model if role == "answer" and not model else ""
        key = s.api_key(self.provider)
        if not key:
            raise LLMError(f"No API key for {self.provider}: set {self.provider.upper()}_API_KEY")
        self.max_retries = max_retries
        self.http = httpx.Client(
            base_url=PROVIDER_URLS[self.provider],
            headers={"Authorization": f"Bearer {key}"},
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
        overloaded = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self.http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                if attempt == self.max_retries:
                    raise LLMError(f"network error: {error}") from error
                time.sleep(2**attempt)
                continue
            if response.status_code == 503:
                overloaded += 1
                if overloaded >= OVERLOADED_TRIES_BEFORE_FALLBACK and self.fallback_model:
                    body["model"] = self.fallback_model
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries:
                    raise LLMError(
                        f"{self.provider} {response.status_code} after {attempt + 1} attempts: "
                        f"{response.text[:200]}"
                    )
                time.sleep(retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise LLMError(f"{self.provider} {response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage") or {}
            return Completion(
                text=data["choices"][0]["message"]["content"] or "",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                seconds=round(time.perf_counter() - started, 2),
                model=body["model"],
            )
        raise LLMError("unreachable")

    def close(self) -> None:
        self.http.close()


def retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    try:
        return min(float(header) + 1, 90.0) if header else min(2**attempt, 60)
    except ValueError:
        return min(2**attempt, 60)
