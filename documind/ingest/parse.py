"""Parse the RBI PDFs in data/raw into clean, structured text in data/processed.

    uv run python -m documind.ingest.parse

For each document this writes data/processed/<doc_id>.json: a list of paragraphs, each with
its page, chapter, heading and paragraph number, so answers can cite "Chapter II, para 12,
page 7". A summary of every document is written to data/processed/report.csv.

Cleaning steps, in order:
  1. Read text spans with their font size and superscript flag (PyMuPDF "dict" mode).
  2. Drop superscript spans: footnote markers glued to words ("8acquisition").
  3. Drop footnotes: lines in a smaller font than the body text, below the body on the page.
  4. Drop lines repeated at the top or bottom of most pages (headers, footers), page numbers,
     table-of-contents lines and non-Latin letterhead text.
  5. Group the remaining lines into paragraphs and track the chapter / heading / paragraph.
"""

import csv
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from documind.config import get_settings

# --- Structure patterns -------------------------------------------------------------------

CHAPTER_RE = re.compile(r"^(Chapter|CHAPTER)\s*[-–]?\s*([IVXLC]+|\d+)\b\s*[-–—:.]?\s*(.*)$")
# "Annex II", "Annexure - 1", "Annex I Reporting format ..." but not "Annex III along with ..."
ANNEX_RE = re.compile(
    r"^(Annex(?:ure)?|Appendix|Schedule)\s*[-–—]?\s*([IVXLC]+|\d+)\s*[-–—:.]?\s*([A-Z(\[].*)?$"
)
PART_RE = re.compile(r"^(Part|PART)\s+([A-Z]|[IVXLC]+|\d+)\b\s*[-–—:.]?\s*([A-Z(].*)?$")
# "A. Short Title and Commencement", "B.2 Review by the Audit Committee of the Board"
HEADING_RE = re.compile(r"^([A-Z]\.(?:\d{1,2}\.?)*)\s+([A-Z(].{2,90})$")
PARA_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.?\s+(\S.*)$")  # "12. ", "5.2 ", "2.1.3 "
SUBPARA_RE = re.compile(r"^\(([a-z]{1,3}|\d{1,3}|[ivxl]{1,6})\)\s+(\S.*)$")  # "(13) ", "(ii) "

LONE_NUMBER_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})\.$")  # "2." with text on next line

TOC_RE = re.compile(r"(\.\s*){5,}\s*\d+\s*$")  # "Chapter I – Preliminary ........ 2"
TOC_TITLE_RE = re.compile(r"^(table of contents|contents|index)$", re.I)
PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,3}(\s*(of|/)\s*\d{1,3})?\s*$", re.I)

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def chapter_number(token: str) -> int:
    if token.isdigit():
        return int(token)
    total = 0
    for i, ch in enumerate(token):
        value = ROMAN[ch]
        total += -value if i + 1 < len(token) and ROMAN[token[i + 1]] > value else value
    return total


@dataclass
class Line:
    text: str
    size: float
    y0: float
    y1: float
    page: int


@dataclass
class Paragraph:
    text: str
    page_start: int
    page_end: int
    chapter: str = ""
    heading: str = ""
    para: str = ""
    kind: str = "text"  # text | heading


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    topic: str
    issued: str
    source_url: str
    pages: int
    paragraphs: list[Paragraph] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --- Reading lines ------------------------------------------------------------------------


def mostly_non_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) < 0.5


def read_lines(pdf: pymupdf.Document) -> tuple[list[Line], dict]:
    """Return text lines with font size and position, without superscript markers."""
    lines: list[Line] = []
    superscripts = 0
    for page_index, page in enumerate(pdf):
        for block in page.get_text("dict")["blocks"]:
            for raw in block.get("lines", []):
                spans = []
                for span in raw["spans"]:
                    if span["flags"] & 1:  # superscript: footnote marker
                        superscripts += 1
                        continue
                    spans.append(span)
                text = "".join(s["text"] for s in spans)
                text = re.sub(r"\s+", " ", text).strip()
                if not text:
                    continue
                size = max(s["size"] for s in spans if s["text"].strip()) if spans else 0
                x0, y0, x1, y1 = raw["bbox"]
                lines.append(Line(text, round(size, 1), y0, y1, page_index + 1))
    return lines, {"superscripts_removed": superscripts}


def merge_same_row(lines: list[Line]) -> list[Line]:
    """PyMuPDF sometimes splits one visual line into pieces; join pieces on the same row."""
    merged: list[Line] = []
    for line in lines:
        prev = merged[-1] if merged else None
        if prev and prev.page == line.page and abs(prev.y0 - line.y0) < 2:
            prev.text = f"{prev.text} {line.text}"
            prev.size = max(prev.size, line.size)
        else:
            merged.append(Line(**asdict(line)))
    return merged


def body_font_size(lines: list[Line]) -> float:
    counts = Counter()
    for line in lines:
        counts[line.size] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 0


def normalise_repeated(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower()).strip()


def toc_pages(by_page: dict[int, list[Line]]) -> set[int]:
    """Pages holding a table of contents: a page with a 'Contents'/'Index' title, plus
    following pages that are still made of short entries."""
    found: set[int] = set()
    for page in sorted(by_page):
        page_lines = by_page[page]
        if any(TOC_TITLE_RE.match(line.text) for line in page_lines[:6]):
            found.add(page)
        elif page - 1 in found:
            short = sum(len(line.text) < 70 for line in page_lines)
            entries = sum(
                bool(
                    TOC_RE.search(line.text)
                    or CHAPTER_RE.match(line.text)
                    or ANNEX_RE.match(line.text)
                    or LONE_NUMBER_RE.match(line.text)
                )
                for line in page_lines
            )
            if page_lines and short / len(page_lines) > 0.8 and entries >= 3:
                found.add(page)
    return found


def remove_noise(lines: list[Line], pages: int) -> tuple[list[Line], dict]:
    body = body_font_size(lines)
    stats = Counter()

    # Headers / footers: the same text (digits ignored) at the top or bottom of many pages.
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page, []).append(line)
    edge_counts = Counter()
    for page_lines in by_page.values():
        for line in page_lines[:2] + page_lines[-2:]:
            edge_counts[normalise_repeated(line.text)] += 1
    repeated = {t for t, n in edge_counts.items() if pages >= 4 and n >= max(3, pages * 0.5)}
    contents = toc_pages(by_page)

    kept: list[Line] = []
    for page, page_lines in by_page.items():
        if page in contents:
            stats["toc_lines"] += len(page_lines)
            continue
        # Footnotes start at the first small-font line after which only small lines follow.
        footnote_start = len(page_lines)
        for i in range(len(page_lines) - 1, -1, -1):
            if page_lines[i].size and page_lines[i].size < body - 1.0:
                footnote_start = i
            elif not PAGE_NUMBER_RE.match(page_lines[i].text):
                break
        for i, line in enumerate(page_lines):
            text = line.text
            if i >= footnote_start:
                stats["footnote_lines"] += 1
            elif normalise_repeated(text) in repeated:
                stats["header_footer_lines"] += 1
            elif PAGE_NUMBER_RE.match(text):
                stats["page_number_lines"] += 1
            elif TOC_RE.search(text):
                stats["toc_lines"] += 1
            elif mostly_non_latin(text):
                stats["non_latin_lines"] += 1
            else:
                kept.append(line)
    stats["body_font_size"] = body
    return kept, dict(stats)


# --- Building paragraphs ------------------------------------------------------------------


def join_text(left: str, right: str) -> str:
    if left.endswith("-") and right[:1].islower():  # word broken across lines
        return left[:-1] + right
    return f"{left} {right}"


def build_paragraphs(lines: list[Line]) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    chapter = heading = para = ""
    last_chapter = 0
    pending_number = ""  # "2." on its own line, waiting for its text
    current: Paragraph | None = None

    def start(text: str, page: int, kind: str = "text") -> Paragraph:
        p = Paragraph(text, page, page, chapter, heading, para, kind)
        paragraphs.append(p)
        return p

    for line in lines:
        text = line.text
        if pending_number:
            text = f"{pending_number}. {text}"
            pending_number = ""
        elif m := LONE_NUMBER_RE.match(text):
            pending_number = m.group(1)
            continue

        # Continuation of a heading split over two lines: "... for resident" / "individuals".
        if (
            current is not None
            and current.kind == "heading"
            and current.page_end == line.page
            and text[:1].islower()
        ):
            was_chapter = current.text == chapter
            current.text = join_text(current.text, text)
            if was_chapter:
                chapter = current.chapter = current.text
            else:
                heading = current.heading = current.text
            continue

        if m := CHAPTER_RE.match(text):
            number = chapter_number(m.group(2))
            title = m.group(3).strip()
            # Accept only the next chapter in sequence, so cross-references that happen to
            # start a line ("Chapter II of these Directions ...") stay in the text.
            if number == last_chapter + 1 and not title[:1].islower():
                last_chapter = number
                chapter = f"Chapter {m.group(2)} – {title}" if title else f"Chapter {m.group(2)} –"
                heading = para = ""
                current = start(chapter, line.page, "heading")
                continue
        if (
            current is not None
            and current.kind == "heading"
            and chapter.endswith("–")
            and len(text) < 100
            and not PARA_RE.match(text)
        ):
            # Chapter title on the line after "Chapter I".
            chapter = current.text = current.chapter = f"{chapter} {text}"
            continue
        if chapter.endswith("–"):  # untitled chapter
            chapter = chapter[:-2]
            if current is not None and current.kind == "heading":
                current.text = current.chapter = chapter
        if ANNEX_RE.match(text) and len(text) < 80 and not text.endswith((".", ",", ";")):
            chapter = text
            heading = para = ""
            current = start(text, line.page, "heading")
            continue
        if PART_RE.match(text) and len(text) < 120:
            heading, para = text, ""
            last_chapter = 0  # chapter numbering may restart in each part
            current = start(text, line.page, "heading")
            continue
        if (m := HEADING_RE.match(text)) and not text.rstrip().endswith((",", ";")):
            heading, para = text, ""
            current = start(text, line.page, "heading")
            continue
        if m := PARA_RE.match(text):
            para = m.group(1)
            current = start(text, line.page)
            continue
        if SUBPARA_RE.match(text) or current is None or current.kind == "heading":
            current = start(text, line.page)
            continue
        current.text = join_text(current.text, text)
        current.page_end = line.page

    # A heading split over two lines arrives as heading + one short text line; keep both.
    return [p for p in paragraphs if p.text.strip()]


# --- Whole document -----------------------------------------------------------------------


def parse_pdf(path: Path, meta: dict) -> ParsedDocument:
    with pymupdf.open(path) as pdf:
        lines, stats = read_lines(pdf)
        pages = pdf.page_count
    lines = merge_same_row(lines)
    lines, noise_stats = remove_noise(lines, pages)
    paragraphs = build_paragraphs(lines)
    text_chars = sum(len(p.text) for p in paragraphs)
    stats.update(noise_stats)
    stats.update(
        paragraphs=len(paragraphs),
        chapters=len({p.chapter for p in paragraphs if p.chapter}),
        numbered_paras=len({(p.chapter, p.para) for p in paragraphs if p.para}),
        chars=text_chars,
        chars_per_page=round(text_chars / pages) if pages else 0,
    )
    return ParsedDocument(
        doc_id=meta["doc_id"],
        title=meta["title"],
        topic=meta["topic"],
        issued=meta["issued"],
        source_url=meta["rbi_page"],
        pages=pages,
        paragraphs=paragraphs,
        stats=stats,
    )


def warnings_for(doc: ParsedDocument) -> list[str]:
    warnings = []
    if doc.stats["chars_per_page"] < 500:
        warnings.append("little text per page (scanned or mostly tables?)")
    if doc.stats["numbered_paras"] == 0:
        warnings.append("no numbered paragraphs found")
    return warnings


def main() -> int:
    settings = get_settings()
    raw_dir = settings.data_dir / "raw"
    out_dir = settings.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (settings.data_dir / "documents.csv").open(encoding="utf-8", newline="") as f:
        documents = list(csv.DictReader(f))

    report = []
    for meta in documents:
        path = raw_dir / f"{meta['doc_id']}.pdf"
        if not path.exists():
            print(f"{meta['doc_id']}: missing, run documind.ingest.download first")
            continue
        doc = parse_pdf(path, meta)
        (out_dir / f"{doc.doc_id}.json").write_text(
            json.dumps(asdict(doc), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        warnings = warnings_for(doc)
        s = doc.stats
        report.append(
            {
                "doc_id": doc.doc_id,
                "pages": doc.pages,
                "paragraphs": s["paragraphs"],
                "chapters": s["chapters"],
                "numbered_paras": s["numbered_paras"],
                "chars": s["chars"],
                "footnote_lines": s.get("footnote_lines", 0),
                "header_footer_lines": s.get("header_footer_lines", 0),
                "superscripts_removed": s["superscripts_removed"],
                "warnings": "; ".join(warnings),
            }
        )
        flag = f"  ⚠ {'; '.join(warnings)}" if warnings else ""
        print(
            f"{doc.doc_id:<10} {doc.pages:>3} pages {s['paragraphs']:>5} paras "
            f"{s['chapters']:>3} chapters {s['chars']:>8,} chars{flag}"
        )

    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    total_pages = sum(r["pages"] for r in report)
    total_chars = sum(r["chars"] for r in report)
    flagged = sum(1 for r in report if r["warnings"])
    print(
        f"\n{len(report)} documents, {total_pages:,} pages, {total_chars:,} characters "
        f"(~{total_chars // 4:,} tokens). {flagged} with warnings."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
