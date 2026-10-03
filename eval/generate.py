"""Draft test questions for human review.

    uv run python -m eval.generate                 # all three kinds
    uv run python -m eval.generate --kinds single  # only one kind

Writes eval/candidates.jsonl (status "pending"). Review them with:
    uv run streamlit run ui/review.py

- single:       one passage -> one question, answer and verbatim evidence quote.
- multi:        two passages (same topic) -> a question that needs both.
- unanswerable: hand-written questions the documents shouldn't answer (out of scope, false
                premise, too specific). Each is run through DocuMind; if it finds an answer,
                the record is flagged so the reviewer checks it.
Quotes the model didn't copy exactly from the passage are rejected automatically.
"""

import argparse
import json
import random
import re
import sys

from documind.chunking.chunkers import load_parsed, short_title
from documind.config import get_settings
from documind.generation.llm import LLMClient, LLMError
from eval.testset import (
    CANDIDATES_PATH,
    EVAL_DIR,
    Evidence,
    Record,
    load_records,
    locate_quote,
    normalise,
    save_records,
)

SEED = 42
PER_DOC = 3
MULTI_COUNT = 30
PROGRESS_PATH = EVAL_DIR / ".generate_progress.json"

QTYPES = "numeric | yes_no | procedure | definition | condition | factual"

SINGLE_PROMPT = """You write test questions for a question-answering system over Reserve Bank \
of India (RBI) regulations. Below is a passage from one RBI document.

Document: {title}
Section: {section}
Passage:
\"\"\"
{text}
\"\"\"

Write ONE question that a bank customer, compliance officer or student might realistically \
ask, whose answer is clearly and completely stated in this passage.

Rules:
- Ask in your own words, the way a person would. Do not copy long phrases from the passage.
- The question must make sense on its own, without seeing the passage: name the product or \
scheme (for example "credit card", "LRS", "a commercial bank", "KYC").
- Never say "according to the passage" or "this document".
- Prefer specific facts: limits, amounts, time periods, conditions, who is allowed to do what.
- "answer": 1-2 sentences.
- "evidence_quote": copy EXACTLY, character for character, the one or two consecutive \
sentences from the passage that contain the answer (at most 60 words).
- If the passage has nothing worth asking about (only repeals, references to other circulars, \
addresses or signatures), reply {{"skip": true}}.

Reply with JSON only:
{{"question": "...", "answer": "...", "evidence_quote": "...", "question_type": "{qtypes}"}}"""

MULTI_PROMPT = """You write HARD test questions for a question-answering system over Reserve \
Bank of India (RBI) regulations. Below are two passages.

Passage A ({title_a} / {section_a}):
\"\"\"
{text_a}
\"\"\"

Passage B ({title_b} / {section_b}):
\"\"\"
{text_b}
\"\"\"

Write ONE realistic question whose complete answer needs facts from BOTH passages, for \
example a comparison, two conditions that both apply, or a rule plus its exception.

Rules:
- Ask in your own words; the question must make sense without seeing the passages.
- "answer": 1-3 sentences combining both passages.
- "evidence_a" and "evidence_b": copy EXACTLY the sentence from each passage that the answer \
needs (at most 50 words each).
- If no natural question needs both passages, reply {{"skip": true}}.

Reply with JSON only:
{{"question": "...", "answer": "...", "evidence_a": "...", "evidence_b": "..."}}"""

# Hand-written. Grouped by why the documents shouldn't answer them.
UNANSWERABLE = {
    "out of scope: monetary policy": [
        "What is the current repo rate set by the RBI?",
        "When is the next RBI Monetary Policy Committee meeting?",
        "What inflation target has the government set for the RBI?",
    ],
    "out of scope: other regulators": [
        "What is the minimum amount for a mutual fund SIP?",
        "How much tax is deducted on fixed deposit interest?",
        "What is the lock-in period for an ELSS mutual fund?",
        "What are the SEBI rules for an IPO allotment?",
        "How much can I invest in the Public Provident Fund each year?",
    ],
    "out of scope: entities not covered": [
        "What is the maximum loan-to-value ratio for gold loans given by NBFCs?",
        "What capital must a small finance bank hold to get a licence?",
        "What are the rules for a co-operative bank to open a new branch?",
        "What is the minimum net owned fund for a housing finance company?",
    ],
    "too specific: names, dates, products": [
        "What penalty did the RBI impose on HDFC Bank in 2025?",
        "What interest rate does SBI pay on a one-year fixed deposit?",
        "Which banks have been classified as domestic systemically important banks this year?",
        "What is the annual fee of the ICICI Bank Coral credit card?",
        "How many credit cards were issued in India last year?",
    ],
    "false premise": [
        "Why did the RBI ban credit cards for students under 21?",
        "What is the penalty for holding more than two savings accounts at one bank?",
        "Why does the RBI require a minimum balance of ₹10,000 in every savings account?",
        "Since when has the LRS allowed remittances for buying lottery tickets abroad?",
        "What is the RBI's daily limit of ₹5,000 on debit card ATM withdrawals?",
    ],
    "future or opinion": [
        "Will the RBI raise the LRS limit next year?",
        "Is it better to take a personal loan or use a credit card for a big purchase?",
        "Which bank offers the best credit card for travel?",
        "Should I invest in foreign stocks under LRS right now?",
    ],
}

# Hand-written hard cases found while testing. Quotes are matched against the parsed text.
HANDWRITTEN = [
    {
        "question": "Can a bank charge fees on a transaction the customer reported as fraud?",
        "answer": "No. No charges can be levied on transactions disputed as fraud by the "
        "cardholder until the dispute is resolved.",
        "doc_id": "rbi-13155",
        "quote": "No charges shall be levied on transactions disputed as ‘fraud’ by the cardholder "
        "until the dispute is resolved.",
        "qtype": "yes_no",
    },
]


# --- Passages -----------------------------------------------------------------------------


def passages(doc: dict, min_words: int = 80, max_words: int = 320) -> list[dict]:
    """Runs of consecutive paragraphs under one heading, sized for a question."""
    out, current = [], None
    for p in doc["paragraphs"]:
        if p["kind"] != "text":
            continue
        key = (p["chapter"], p["heading"])
        words = len(p["text"].split())
        if current and current["key"] == key and current["words"] + words <= max_words:
            current["paras"].append(p)
            current["words"] += words
            continue
        if current and current["words"] >= min_words:
            out.append(current)
        current = {"key": key, "paras": [p], "words": words}
    if current and current["words"] >= min_words:
        out.append(current)
    for passage in out:
        passage["text"] = "\n".join(p["text"] for p in passage["paras"])
        passage["section"] = " > ".join(x for x in passage["key"] if x) or "(no section)"
    return out


def useful(passage: dict) -> bool:
    text = passage["text"].lower()
    boring = ("repeal" in passage["section"].lower()) or text.count("circular") > 4
    return not boring and (bool(re.search(r"\d", text)) or "shall" in text)


def pick_passages(doc: dict, n: int, rng: random.Random) -> list[dict]:
    """Prefer passages from different chapters, so questions cover the whole document."""
    pool = [p for p in passages(doc) if useful(p)]
    rng.shuffle(pool)
    chosen, chapters = [], set()
    for p in pool:
        if p["key"][0] not in chapters:
            chosen.append(p)
            chapters.add(p["key"][0])
        if len(chosen) == n:
            return chosen
    return chosen + [p for p in pool if p not in chosen][: n - len(chosen)]


def evidence_for(doc: dict, passage: dict, quote: str) -> Evidence | None:
    exact = locate_quote(quote, passage["text"])
    if exact is None:
        return None
    for p in passage["paras"]:  # the paragraph where the quote starts gives para and page
        if exact[:60] in normalise(p["text"]):
            return Evidence(doc["doc_id"], exact, p["para"], p["page_start"])
    first = passage["paras"][0]
    return Evidence(doc["doc_id"], exact, first["para"], first["page_start"])


def ask_json(llm: LLMClient, prompt: str) -> dict | None:
    """The model's JSON reply, or None if it isn't usable. LLMError (rate limits, outages)
    is raised so the caller can leave the item unfinished and retry it on the next run."""
    reply = llm.chat([{"role": "user", "content": prompt}], json_mode=True, temperature=0.7)
    try:
        return json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        print("    reply was not valid JSON, skipped")
        return None


# --- Saving as we go ----------------------------------------------------------------------


class RateLimited(RuntimeError):
    pass


class Store:
    """Saves after every record and remembers which items were already tried, so a crash or
    rate limit loses at most one item and the next run continues where this one stopped."""

    MAX_FAILURES_IN_A_ROW = 3

    def __init__(self, fresh: bool):
        if fresh:
            CANDIDATES_PATH.unlink(missing_ok=True)
            PROGRESS_PATH.unlink(missing_ok=True)
        self.records = load_records(CANDIDATES_PATH)
        self.done: set[str] = (
            set(json.loads(PROGRESS_PATH.read_text(encoding="utf-8")))
            if PROGRESS_PATH.exists()
            else set()
        )
        self.failures_in_a_row = 0

    def next_id(self, prefix: str) -> str:
        return f"{prefix}{sum(r.id.startswith(prefix) for r in self.records) + 1:03d}"

    def count(self, kind: str) -> int:
        return sum(r.kind == kind for r in self.records)

    def add(self, record: Record) -> None:
        self.records.append(record)
        save_records(self.records, CANDIDATES_PATH)

    def finish(self, key: str) -> None:
        self.done.add(key)
        self.failures_in_a_row = 0
        PROGRESS_PATH.write_text(json.dumps(sorted(self.done)), encoding="utf-8")

    def failed(self, key: str, error: Exception) -> None:
        self.failures_in_a_row += 1
        print(f"    {key}: {error} (will retry on the next run)")
        if self.failures_in_a_row >= self.MAX_FAILURES_IN_A_ROW:
            raise RateLimited(
                "The LLM keeps refusing requests (rate limit). Everything so far is saved; "
                "run the same command again later to continue."
            )


# --- Generators ---------------------------------------------------------------------------


def generate_single(docs: list[dict], llm: LLMClient, rng: random.Random, store: Store) -> None:
    rejected = 0
    for doc in docs:
        title = short_title(doc["title"])
        # Always pick, even for finished docs, so the random sequence is the same on resume.
        for i, passage in enumerate(pick_passages(doc, PER_DOC, rng)):
            key = f"single:{doc['doc_id']}:{i}"
            if key in store.done:
                continue
            prompt = SINGLE_PROMPT.format(
                title=title, section=passage["section"], text=passage["text"], qtypes=QTYPES
            )
            try:
                data = ask_json(llm, prompt)
            except LLMError as error:
                store.failed(key, error)
                continue
            usable = data is not None and not data.get("skip")
            evidence = (
                evidence_for(doc, passage, data.get("evidence_quote", "")) if usable else None
            )
            if usable and evidence is None:
                rejected += 1
            if evidence:
                store.add(
                    Record(
                        id=store.next_id("s"),
                        question=data["question"].strip(),
                        answer=data["answer"].strip(),
                        kind="single",
                        qtype=data.get("question_type", ""),
                        evidence=[evidence],
                    )
                )
            store.finish(key)
        print(f"  single: {doc['doc_id']} done, {store.count('single')} questions so far")
    print(f"  single: {rejected} rejected this run (quote not found in passage)")


def generate_multi(docs: list[dict], llm: LLMClient, rng: random.Random, store: Store) -> None:
    by_topic: dict[str, list[dict]] = {}
    for doc in docs:
        by_topic.setdefault(doc["topic"], []).append(doc)
    for attempt in range(MULTI_COUNT * 3):
        if store.count("multi") >= MULTI_COUNT:
            break
        # Draw the pair every time, so the random sequence is the same on resume.
        topic_docs = rng.choice(list(by_topic.values()))
        doc_a = rng.choice(topic_docs)
        doc_b = rng.choice(topic_docs) if rng.random() < 0.6 else doc_a  # often across docs
        pa, pb = pick_passages(doc_a, 1, rng), pick_passages(doc_b, 1, rng)
        key = f"multi:{attempt}"
        if key in store.done or not pa or not pb or pa[0]["text"] == pb[0]["text"]:
            continue
        pa, pb = pa[0], pb[0]
        prompt = MULTI_PROMPT.format(
            title_a=short_title(doc_a["title"]),
            section_a=pa["section"],
            text_a=pa["text"],
            title_b=short_title(doc_b["title"]),
            section_b=pb["section"],
            text_b=pb["text"],
        )
        try:
            data = ask_json(llm, prompt)
        except LLMError as error:
            store.failed(key, error)
            continue
        if data and not data.get("skip"):
            ev_a = evidence_for(doc_a, pa, data.get("evidence_a", ""))
            ev_b = evidence_for(doc_b, pb, data.get("evidence_b", ""))
            if ev_a and ev_b:
                store.add(
                    Record(
                        id=store.next_id("m"),
                        question=data["question"].strip(),
                        answer=data["answer"].strip(),
                        kind="multi",
                        qtype="multi_hop",
                        evidence=[ev_a, ev_b],
                    )
                )
                print(f"  multi: {store.count('multi')}/{MULTI_COUNT}")
        store.finish(key)


def generate_unanswerable(store: Store, llm: LLMClient) -> None:
    from documind.generation.answer import Answerer

    answerer = Answerer(llm=llm)
    try:
        for reason, questions in UNANSWERABLE.items():
            for question in questions:
                key = f"unanswerable:{question}"
                if key in store.done:
                    continue
                try:
                    result = answerer.answer(question)
                except LLMError as error:
                    store.failed(key, error)
                    continue
                note = f"reason: {reason}"
                if result.found:
                    note += f" | CHECK: DocuMind answered: {result.answer[:300]}"
                store.add(
                    Record(
                        id=store.next_id("u"),
                        question=question,
                        answer="",
                        kind="unanswerable",
                        qtype=reason.split(":")[0],
                        origin="handwritten",
                        note=note,
                    )
                )
                store.finish(key)
                flag = " (DocuMind answered it: flagged for review)" if result.found else ""
                print(f"  unanswerable: {store.count('unanswerable')}{flag}")
    finally:
        answerer.close()


def generate_handwritten(docs: list[dict], store: Store) -> None:
    by_id = {d["doc_id"]: d for d in docs}
    for i, item in enumerate(HANDWRITTEN, 1):
        key = f"handwritten:{i}"
        if key in store.done:
            continue
        doc = by_id[item["doc_id"]]
        for p in doc["paragraphs"]:
            if exact := locate_quote(item["quote"], p["text"]):
                store.add(
                    Record(
                        id=f"h{i:03d}",
                        question=item["question"],
                        answer=item["answer"],
                        kind="single",
                        qtype=item["qtype"],
                        evidence=[Evidence(doc["doc_id"], exact, p["para"], p["page_start"])],
                        origin="handwritten",
                    )
                )
                break
        else:
            print(f"  handwritten {i}: quote not found, skipped")
        store.finish(key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft test questions for review")
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["single", "multi", "unanswerable"],
        choices=["single", "multi", "unanswerable"],
    )
    parser.add_argument("--fresh", action="store_true", help="discard saved progress and restart")
    parser.add_argument("--model", default=None, help="LLM to draft with (default: LLM_MODEL)")
    args = parser.parse_args()

    settings = get_settings()
    docs = load_parsed(settings.data_dir / "processed")
    rng = random.Random(SEED)
    store = Store(fresh=args.fresh)
    llm = LLMClient(model=args.model)
    try:
        if "single" in args.kinds:
            generate_handwritten(docs, store)
            generate_single(docs, llm, rng, store)
        if "multi" in args.kinds:
            generate_multi(docs, llm, rng, store)
        if "unanswerable" in args.kinds:
            generate_unanswerable(store, llm)
    except RateLimited as stop:
        print(f"\n{stop}")
        return 2
    finally:
        llm.close()
        counts = {k: store.count(k) for k in ("single", "multi", "unanswerable")}
        print(f"\n{len(store.records)} candidates saved in {CANDIDATES_PATH.name}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
