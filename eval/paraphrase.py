"""Make a paraphrased copy of the test set: same evidence, everyday wording.

    uv run python -m eval.paraphrase

The test questions were drafted from the passages, so they share vocabulary with them, which
flatters keyword search. Real users don't quote regulations. This rewrites each answerable
question the way a customer or a junior employee would ask it, avoiding the passage's exact
terms where natural, and keeps the same id, answer and evidence. Writes
eval/dataset_paraphrased.jsonl; saves after every question and resumes if interrupted.
Unanswerable questions are already in plain language and are copied unchanged.
"""

import json
import re
import sys

from documind.generation.llm import DailyLimitError, LLMClient, LLMError
from eval.testset import DATASET_PATH, EVAL_DIR, load_records, save_records

PARAPHRASED_PATH = EVAL_DIR / "dataset_paraphrased.jsonl"

PROMPT = """Rewrite this question the way a bank customer or a new bank employee would \
naturally ask it, in everyday words.

Rules:
- Keep exactly the same meaning and the same information need.
- Avoid the formal or technical terms of the original where an everyday phrase exists \
(for example "fees" instead of "charges levied", "wallet" instead of "PPI"), but keep \
proper names a person would know (LRS, KYC, UPI, NRE, credit card).
- One question, no preamble.

Original: {question}

Reply with JSON only: {{"question": "..."}}"""


def main() -> int:
    records = load_records(DATASET_PATH)
    done = {r.id: r for r in load_records(PARAPHRASED_PATH)}
    llm = LLMClient()
    try:
        for record in records:
            if record.id in done:
                continue
            if record.kind == "unanswerable":
                done[record.id] = record
                continue
            try:
                reply = llm.chat(
                    [{"role": "user", "content": PROMPT.format(question=record.question)}],
                    json_mode=True,
                    temperature=0.7,
                    max_tokens=200,
                )
                match = re.search(r"\{.*\}", reply.text, re.S)
                question = json.loads(match.group(0))["question"].strip() if match else ""
            except DailyLimitError as error:
                print(f"Stopped: {error}\nRun again later to continue.")
                break
            except (LLMError, json.JSONDecodeError, KeyError) as error:
                print(f"  {record.id}: {error} (will retry next run)")
                continue
            if not question:
                continue
            done[record.id] = record.__class__(
                **{
                    **record.__dict__,
                    "question": question,
                    "note": f"paraphrase of: {record.question}",
                }
            )
            save_records([done[r.id] for r in records if r.id in done], PARAPHRASED_PATH)
            print(f"  {record.id}: {question}")
    finally:
        llm.close()
    save_records([done[r.id] for r in records if r.id in done], PARAPHRASED_PATH)
    print(f"\n{len(done)} of {len(records)} questions in {PARAPHRASED_PATH.name}")
    return 0 if len(done) == len(records) else 2


if __name__ == "__main__":
    sys.exit(main())
