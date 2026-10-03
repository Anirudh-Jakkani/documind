from documind.chunking.chunkers import (
    chunk_document,
    short_title,
    split_sentences,
    split_to_fit,
)


def words(text: str) -> int:
    return len(text.split())


def para(text, chapter="Chapter I – Preliminary", heading="A. Scope", number="1", page=1):
    return {
        "text": text,
        "page_start": page,
        "page_end": page,
        "chapter": chapter,
        "heading": heading,
        "para": number,
        "kind": "text",
    }


def doc(*paragraphs):
    return {
        "doc_id": "rbi-1",
        "title": "Reserve Bank of India (Commercial Banks – Credit Cards) Directions, 2025 "
        "(Updated as on October 1, 2026)",
        "topic": "Banking",
        "source_url": "https://rbi.org.in/x",
        "paragraphs": list(paragraphs),
    }


def test_short_title():
    assert short_title(doc()["title"]) == "Commercial Banks – Credit Cards"
    assert (
        short_title(
            "Master Direction - Liberalised Remittance Scheme (LRS) "
            "(Updated as on September 06, 2024)"
        )
        == "Liberalised Remittance Scheme (LRS)"
    )


def test_sentences_not_split_after_abbreviations():
    text = "Refer to Circular No. 12 dated 2024. Banks shall comply."
    assert split_sentences(text) == ["Refer to Circular No. 12 dated 2024.", "Banks shall comply."]


def test_split_to_fit_respects_limit():
    text = " ".join(f"Sentence number {i} has five words." for i in range(20))
    parts = split_to_fit(text, 12, words)
    assert all(words(p) <= 12 for p in parts)
    assert " ".join(parts) == text


def test_section_chunks_never_cross_headings_and_carry_header():
    d = doc(
        para("1. " + "alpha " * 30, heading="A. Scope", number="1"),
        para("2. " + "beta " * 30, heading="B. Definitions", number="2", page=2),
    )
    chunks = chunk_document(d, "section", words, size=200, overlap=0, min_tokens=10)
    assert len(chunks) == 2
    assert (
        chunks[0].header == "Commercial Banks – Credit Cards > Chapter I – Preliminary > A. Scope"
    )
    assert "beta" not in chunks[0].text
    assert chunks[1].paras == ["2"] and chunks[1].page_start == 2
    assert chunks[1].index_text.startswith(chunks[1].header)


def test_section_breaks_at_new_paragraph_once_big_enough():
    d = doc(*[para(f"{i}. " + "word " * 20, number=str(i)) for i in range(1, 7)])
    chunks = chunk_document(d, "section", words, size=500, overlap=0, min_tokens=40)
    assert [c.paras for c in chunks] == [["1", "2"], ["3", "4"], ["5", "6"]]


def test_fixed_windows_overlap_and_fit():
    d = doc(para(" ".join(f"w{i}" for i in range(100))))
    chunks = chunk_document(d, "fixed", words, size=30, overlap=10, min_tokens=0)
    assert all(c.n_tokens <= 30 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]
    assert chunks[-1].text.split()[-1] == "w99"


def test_recursive_merges_short_paragraphs():
    d = doc(*[para(f"{i}. short text", number=str(i)) for i in range(1, 5)])
    chunks = chunk_document(d, "recursive", words, size=100, overlap=0)
    assert len(chunks) == 1 and chunks[0].paras == ["1", "2", "3", "4"]


def test_citation_format():
    d = doc(para("1. a", number="1", page=3), para("2. b", number="2", page=4))
    chunk = chunk_document(d, "section", words, size=100, overlap=0, min_tokens=50)[0]
    assert chunk.citation() == (
        "Commercial Banks – Credit Cards, Chapter I – Preliminary, paras 1–2, pp. 3–4"
    )
