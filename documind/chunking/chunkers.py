"""Three ways to split parsed documents into chunks, so they can be compared in Phase 6.

- fixed:     sliding windows of N tokens with overlap, ignoring document structure.
- recursive: paragraphs merged up to N tokens; long paragraphs split by sentence, then word.
- section:   like recursive, but never crosses a chapter/heading boundary, prefers to break
             where a numbered paragraph ends, and prefixes each chunk with a context header
             ("Credit Cards and Debit Cards > Chapter II – ... > B.2 Review by ...").

Every chunk keeps its document, chapter, heading, paragraph numbers and pages for citations.
Token counting is passed in, so tests can use a simple word counter instead of a tokenizer.
"""

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

CountTokens = Callable[[str], int]
STRATEGIES = ("fixed", "recursive", "section")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    topic: str
    strategy: str
    text: str
    header: str = ""
    chapter: str = ""
    heading: str = ""
    paras: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    n_tokens: int = 0
    source_url: str = ""

    @property
    def index_text(self) -> str:
        """Text that is embedded and keyword-indexed: the header gives the chunk context."""
        return f"{self.header}\n{self.text}" if self.header else self.text

    def citation(self) -> str:
        parts = [self.doc_title]
        if self.chapter:
            parts.append(self.chapter)
        if self.paras:
            first, last = self.paras[0], self.paras[-1]
            parts.append(f"para {first}" if first == last else f"paras {first}–{last}")
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}–{self.page_end}"
        )
        parts.append(pages)
        return ", ".join(parts)


@dataclass
class Piece:
    """A paragraph, or part of one, with the metadata a chunk needs."""

    text: str
    page_start: int
    page_end: int
    chapter: str
    heading: str
    para: str
    tokens: int


# --- Helpers ------------------------------------------------------------------------------


RBI_TITLE_RE = r"^Reserve Bank of India \((.+)\) (?:Supervisory )?(?:Directions|Guidelines)"


def short_title(title: str) -> str:
    """'Reserve Bank of India (Commercial Banks – Credit Cards ...) Directions, 2025 (Updated
    as on ...)' -> 'Commercial Banks – Credit Cards ...'."""
    title = re.sub(r"\s*\((?:Updated|updated)[^)]*\)\s*$", "", title).strip()
    if m := re.match(RBI_TITLE_RE, title):
        return m.group(1)
    if m := re.match(r"^Reserve Bank of India \[(.+)\] Directions", title):
        return m.group(1)
    title = re.sub(r"^Master Directions?\s*[-–]?\s*(on\s+)?", "", title)
    return re.sub(r",? dated .*$", "", title).strip()


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d‘'\"])")
ABBREVIATIONS = ("No.", "Rs.", "viz.", "i.e.", "e.g.", "etc.", "Sr.", "Para.", "Ltd.", "Co.")


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        if sentences and sentences[-1].endswith(ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return sentences


def split_to_fit(text: str, max_tokens: int, count: CountTokens) -> list[str]:
    """Split text into parts of at most max_tokens: by sentence first, then by words."""
    if count(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        if count(sentence) > max_tokens:  # one very long sentence: split by words
            words, current_words = sentence.split(), []
            for word in words:
                if current_words and count(" ".join(current_words + [word])) > max_tokens:
                    parts.append(" ".join(current_words))
                    current_words = []
                current_words.append(word)
            sentence = " ".join(current_words)
        candidate = f"{current} {sentence}".strip()
        if current and count(candidate) > max_tokens:
            parts.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def top_level(para: str) -> str:
    return para.split(".")[0] if para else ""


def to_pieces(
    doc: dict, max_tokens: int, count: CountTokens, include_headings: bool
) -> list[Piece]:
    pieces: list[Piece] = []
    for p in doc["paragraphs"]:
        if p["kind"] == "heading" and not include_headings:
            continue
        for part in split_to_fit(p["text"], max_tokens, count):
            pieces.append(
                Piece(
                    part,
                    p["page_start"],
                    p["page_end"],
                    p["chapter"],
                    p["heading"],
                    p["para"],
                    count(part),
                )
            )
    return pieces


def make_chunk(
    doc: dict, strategy: str, number: int, pieces: list[Piece], header: str, count: CountTokens
) -> Chunk:
    text = " ".join(p.text for p in pieces)
    paras = []
    for p in pieces:
        if p.para and (not paras or paras[-1] != p.para):
            paras.append(p.para)
    return Chunk(
        chunk_id=f"{doc['doc_id']}:{strategy}:{number:04d}",
        doc_id=doc["doc_id"],
        doc_title=short_title(doc["title"]),
        topic=doc["topic"],
        strategy=strategy,
        text=text,
        header=header,
        chapter=pieces[0].chapter,
        heading=pieces[0].heading,
        paras=paras,
        page_start=min(p.page_start for p in pieces),
        page_end=max(p.page_end for p in pieces),
        n_tokens=count(f"{header}\n{text}" if header else text),
        source_url=doc["source_url"],
    )


# --- Strategies ---------------------------------------------------------------------------


def chunk_fixed(doc: dict, count: CountTokens, size: int, overlap: int) -> list[Chunk]:
    """Sliding windows over the document's words."""
    words: list[tuple[str, Piece]] = []
    for piece in to_pieces(doc, size, count, include_headings=True):
        words.extend((w, piece) for w in piece.text.split())
    if not words:
        return []
    # Token cost per word, estimated per piece, so windows can be sized without re-tokenising.
    cost = [p.tokens / max(1, len(p.text.split())) for _, p in words]

    chunks, start, number = [], 0, 0
    while start < len(words):
        end, total = start, 0.0
        while end < len(words) and total + cost[end] <= size:
            total += cost[end]
            end += 1
        end = max(end, start + 1)
        window = words[start:end]
        # Re-group consecutive words from the same piece so metadata stays accurate.
        pieces: list[Piece] = []
        for word, piece in window:
            if pieces and pieces[-1].para == piece.para and pieces[-1].chapter == piece.chapter:
                pieces[-1].text += f" {word}"
                pieces[-1].page_end = max(pieces[-1].page_end, piece.page_end)
            else:
                pieces.append(
                    Piece(
                        word,
                        piece.page_start,
                        piece.page_end,
                        piece.chapter,
                        piece.heading,
                        piece.para,
                        0,
                    )
                )
        chunks.append(make_chunk(doc, "fixed", number, pieces, "", count))
        number += 1
        if end >= len(words):
            break
        back, back_tokens = end, 0.0
        while back > start + 1 and back_tokens + cost[back - 1] <= overlap:
            back -= 1
            back_tokens += cost[back]
        start = back
    return chunks


def merge_pieces(pieces: Iterable[Piece], size: int, break_on_para: bool, min_tokens: int):
    """Greedily merge pieces up to `size` tokens. With break_on_para, also start a new chunk
    when a new top-level paragraph begins and the current chunk already has min_tokens."""
    group: list[Piece] = []
    tokens = 0
    for piece in pieces:
        new_para = group and top_level(piece.para) != top_level(group[-1].para)
        if group and (
            tokens + piece.tokens > size or (break_on_para and new_para and tokens >= min_tokens)
        ):
            yield group
            group, tokens = [], 0
        group.append(piece)
        tokens += piece.tokens
    if group:
        yield group


def chunk_recursive(doc: dict, count: CountTokens, size: int) -> list[Chunk]:
    pieces = to_pieces(doc, size, count, include_headings=True)
    groups = merge_pieces(pieces, size, break_on_para=False, min_tokens=0)
    return [make_chunk(doc, "recursive", i, g, "", count) for i, g in enumerate(groups)]


def section_header(doc: dict, chapter: str, heading: str) -> str:
    return " > ".join(x for x in (short_title(doc["title"]), chapter, heading) if x)


def chunk_section(doc: dict, count: CountTokens, size: int, min_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    # Consecutive runs of pieces under the same chapter and heading.
    sections: list[tuple[tuple[str, str], list[Piece]]] = []
    # Pieces are split smaller than `size` to leave room for the header.
    for piece in to_pieces(doc, size - 60, count, include_headings=False):
        key = (piece.chapter, piece.heading)
        if not sections or sections[-1][0] != key:
            sections.append((key, []))
        sections[-1][1].append(piece)
    for key, pieces in sections:
        header = section_header(doc, *key)
        budget = size - count(header)
        for group in merge_pieces(pieces, budget, break_on_para=True, min_tokens=min_tokens):
            chunks.append(make_chunk(doc, "section", len(chunks), group, header, count))
    return chunks


def chunk_document(
    doc: dict, strategy: str, count: CountTokens, size: int, overlap: int, min_tokens: int = 150
) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, count, size, overlap)
    if strategy == "recursive":
        return chunk_recursive(doc, count, size)
    if strategy == "section":
        return chunk_section(doc, count, size, min_tokens)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


# --- Loading and saving -------------------------------------------------------------------


def load_parsed(processed_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(processed_dir.glob("rbi-*.json"))
    ]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
