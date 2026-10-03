"""LLM judge: grades an answer's correctness and faithfulness.

The judge uses a different model family from the answering model (see JUDGE_* settings), so
it doesn't favour its own style. It sees only the sources the answer cited, which keeps each
call small and makes "faithfulness" mean "supported by what it cited".
"""

import json
import re

from documind.generation.llm import LLMClient

CORRECTNESS = ("correct", "partial", "incorrect")
FAITHFULNESS = ("supported", "partially_supported", "unsupported")

JUDGE_PROMPT = """You are grading an answer from a question-answering system over Reserve Bank \
of India regulations.

Question: {question}

Reference answer (checked by reviewers): {reference}

System answer: {answer}

Sources the system cited:
{sources}

Grade two things.

1. correctness: compare the system answer with the reference answer.
   - "correct": all key facts of the reference are present and right. Extra correct detail \
is fine. Different wording is fine.
   - "partial": some key facts are right, but something important is missing or wrong.
   - "incorrect": the main fact is wrong or missing.

2. faithfulness: is every factual claim in the system answer supported by the cited sources \
above? Judge only against these sources, not your own knowledge.
   - "supported", "partially_supported" or "unsupported".

Reply with JSON only:
{{"correctness": "...", "faithfulness": "...", "reason": "one short sentence"}}"""


def build_prompt(question: str, reference: str, answer: str, sources: list[dict]) -> str:
    cited = "\n\n".join(f"[{s['n']}] {s['text']}" for s in sources) or "(no sources cited)"
    return JUDGE_PROMPT.format(question=question, reference=reference, answer=answer, sources=cited)


def parse_verdict(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    data = json.loads(match.group(0)) if match else {}
    correctness = str(data.get("correctness", "")).strip().lower()
    faithfulness = str(data.get("faithfulness", "")).strip().lower().replace(" ", "_")
    if correctness not in CORRECTNESS or faithfulness not in FAITHFULNESS:
        raise ValueError(f"unexpected verdict: {text[:200]}")
    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "reason": str(data.get("reason", ""))[:300],
    }


def judge(
    llm: LLMClient, question: str, reference: str, answer: str, sources: list[dict]
) -> tuple[dict, int]:
    """Returns the verdict and the tokens used."""
    completion = llm.chat(
        [{"role": "user", "content": build_prompt(question, reference, answer, sources)}],
        json_mode=True,
        temperature=0.0,
    )
    return parse_verdict(completion.text), completion.prompt_tokens + completion.completion_tokens
