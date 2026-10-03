"""Query rewriting: turn an everyday question into the wording a regulation would use.

Users say "fees", regulations say "charges levied"; users say "wallet", regulations say
"prepaid payment instrument (PPI)". The LLM writes a few alternative search queries, which are
searched alongside the original and fused (see Retriever.search, extra_queries).
"""

import json
import re
from pathlib import Path

from documind.generation.llm import LLMClient

REWRITE_PROMPT = """You help search Reserve Bank of India (RBI) regulations (Master Directions \
for commercial banks, payments, foreign exchange and financial inclusion).

Rewrite the user's question as {n} short search queries that use the formal terms such a \
regulation would use (for example "charges levied" instead of "fees", "prepaid payment \
instrument (PPI)" instead of "wallet", "Liberalised Remittance Scheme" instead of "sending \
money abroad"). Keep each query specific to the question. Do not answer it.

Question: {question}

Reply with JSON only: {{"queries": ["...", "..."]}}"""


class QueryRewriter:
    def __init__(self, llm: LLMClient, n: int = 2, cache_path: Path | None = None):
        self.llm, self.n, self.cache_path = llm, n, cache_path
        self.cache: dict[str, list[str]] = (
            json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_path and cache_path.exists()
            else {}
        )

    def rewrite(self, question: str) -> list[str]:
        if question in self.cache:
            return self.cache[question]
        completion = self.llm.chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(n=self.n, question=question)}],
            json_mode=True,
            max_tokens=300,
        )
        queries = parse_queries(completion.text)[: self.n]
        self.cache[question] = queries
        if self.cache_path:
            self.cache_path.write_text(
                json.dumps(self.cache, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        return queries


def parse_queries(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return []
    try:
        queries = json.loads(match.group(0)).get("queries", [])
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in queries if isinstance(q, str) and q.strip()]
