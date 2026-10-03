"""Test set records, file I/O and evidence matching.

A record's evidence is a verbatim quote from the source document, not a chunk id, so the same
test set can score every chunking strategy: a retrieved chunk is relevant if it contains the
quote (see `chunk_contains`).
"""

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = EVAL_DIR / "candidates.jsonl"
DATASET_PATH = EVAL_DIR / "dataset.jsonl"

KINDS = ("single", "multi", "unanswerable")
STATUSES = ("pending", "approved", "edited", "rejected")


@dataclass
class Evidence:
    doc_id: str
    quote: str
    para: str = ""
    page: int = 0


@dataclass
class Record:
    id: str
    question: str
    answer: str
    kind: str  # single | multi | unanswerable
    qtype: str = ""  # numeric, yes_no, procedure, definition, condition, factual, ...
    evidence: list[Evidence] = field(default_factory=list)
    origin: str = "llm"  # llm | handwritten
    status: str = "pending"
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        data = dict(data)
        data["evidence"] = [Evidence(**e) for e in data.get("evidence", [])]
        return cls(**data)


def load_records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Record.from_dict(json.loads(line)) for line in f if line.strip()]


def save_records(records: list[Record], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic, so a crash never leaves a half-written file


# --- Matching quotes ----------------------------------------------------------------------

_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‑": "-"})


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).lower()
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(quote: str, text: str, min_ratio: float = 0.9) -> str | None:
    """Return the passage's own wording of `quote`: exact if possible, else the closest span
    when the model changed a character or two. None if the quote isn't really in the text."""
    q, t = normalise(quote), normalise(text)
    if not q:
        return None
    if q in t:
        return q
    matcher = SequenceMatcher(None, t, q, autojunk=False)
    match = matcher.find_longest_match(0, len(t), 0, len(q))
    start = max(0, match.a - match.b)
    window = t[start : start + len(q)]
    if SequenceMatcher(None, window, q, autojunk=False).ratio() >= min_ratio:
        return window
    return None


def key_span(quote: str, words: int = 12) -> str:
    """The middle `words` words of a quote. A chunk containing this span holds the evidence,
    even when a fixed-size chunk boundary cuts off the quote's first or last words."""
    tokens = normalise(quote).split()
    if len(tokens) <= words:
        return " ".join(tokens)
    start = (len(tokens) - words) // 2
    return " ".join(tokens[start : start + words])


def longest_run(quote_words: list[str], text: str) -> int:
    """Length of the longest run of consecutive quote words that appears in `text`."""
    best = 0
    for start in range(len(quote_words)):
        if len(quote_words) - start <= best:
            break
        end = start + best + 1  # only longer runs matter
        while end <= len(quote_words) and " ".join(quote_words[start:end]) in text:
            best = end - start
            end += 1
    return best


def is_relevant(chunk_doc_id: str, normalised_text: str, evidence: Evidence) -> bool:
    """A chunk holds the evidence if it contains the quote's middle span (from any document:
    the same rule text sometimes appears in two directions), or, from the same document, at
    least half of the quote as one unbroken run, for when a chunk boundary cuts the quote."""
    if key_span(evidence.quote) in normalised_text:
        return True
    if chunk_doc_id != evidence.doc_id:
        return False
    words = normalise(evidence.quote).split()
    return longest_run(words, normalised_text) >= max(6, (len(words) + 1) // 2)


def chunk_contains(chunk_text: str, evidence: Evidence, chunk_doc_id: str = "") -> bool:
    return is_relevant(chunk_doc_id or evidence.doc_id, normalise(chunk_text), evidence)
