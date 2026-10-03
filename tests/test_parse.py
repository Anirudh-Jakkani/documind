from documind.ingest.parse import (
    Line,
    build_paragraphs,
    chapter_number,
    mostly_non_latin,
    remove_noise,
)


def lines(*texts, page=1, size=12.0):
    return [Line(t, size, i * 10.0, i * 10.0 + 8, page) for i, t in enumerate(texts)]


def test_chapter_numbers():
    assert [chapter_number(t) for t in ["I", "IV", "IX", "XIV", "XVIII", "7"]] == [
        1,
        4,
        9,
        14,
        18,
        7,
    ]


def test_paragraphs_carry_chapter_heading_and_number():
    paras = build_paragraphs(
        lines(
            "Chapter I – Preliminary",
            "A. Short Title and Commencement",
            "1. These Directions shall be called the Reserve Bank of India",
            "(Commercial Banks) Directions, 2025.",
            "2. These Directions shall come into force with immediate effect.",
        )
    )
    text = [p for p in paras if p.kind == "text"]
    assert text[0].chapter == "Chapter I – Preliminary"
    assert text[0].heading == "A. Short Title and Commencement"
    assert text[0].para == "1"
    assert text[0].text.endswith("(Commercial Banks) Directions, 2025.")
    assert text[1].para == "2"


def test_cross_reference_at_line_start_is_not_a_chapter():
    paras = build_paragraphs(
        lines(
            "Chapter I – Preliminary",
            "5. Banks shall follow the instructions in",
            "Chapter II of these Directions, with the approval of its Board.",
            "Chapter II – Board Approved Policies",
        )
    )
    chapters = [p.text for p in paras if p.kind == "heading"]
    assert chapters == ["Chapter I – Preliminary", "Chapter II – Board Approved Policies"]
    assert "Chapter II of these Directions" in paras[1].text


def test_hyphenated_chapter_and_title_on_next_line():
    paras = build_paragraphs(lines("Chapter-I Preliminary", "1. Text."))
    assert paras[0].text == "Chapter I – Preliminary"
    paras = build_paragraphs(lines("Chapter I", "Preliminary", "1. Text."))
    assert paras[0].text == "Chapter I – Preliminary"
    assert paras[1].chapter == "Chapter I – Preliminary"


def test_heading_split_over_two_lines():
    paras = build_paragraphs(
        lines("A. Liberalised Remittance Scheme (LRS) for resident", "individuals", "1. Text.")
    )
    assert paras[0].text == "A. Liberalised Remittance Scheme (LRS) for resident individuals"
    assert paras[1].heading == paras[0].text


def test_sentence_after_heading_is_not_joined_to_it():
    paras = build_paragraphs(
        lines("Schedule III", "stands delegated to the Authorised Dealer banks.", "1. Text.")
    )
    assert paras[0].text == "Schedule III"
    assert paras[1].text == "stands delegated to the Authorised Dealer banks."
    paras = build_paragraphs(lines("1. Text.", "Schedule III stands delegated to the AD banks."))
    assert all(p.kind == "text" for p in paras)


def test_paragraph_number_on_its_own_line():
    paras = build_paragraphs(lines("2.", "Subsequently on April 16, 2024, the RBI placed"))
    assert paras[0].para == "2"
    assert paras[0].text.startswith("2. Subsequently")


def test_annex_reference_in_text_is_not_a_heading():
    paras = build_paragraphs(
        lines("1. Report the items listed in", "Annex III along with a description.", "Annex III")
    )
    assert [p.text for p in paras if p.kind == "heading"] == ["Annex III"]


def test_footnotes_page_numbers_and_toc_are_removed():
    page1 = lines(
        "Table of Contents",
        "Chapter I – Preliminary ........................ 2",
        "Chapter II – Credit Cards ........................ 6",
        page=1,
    )
    page2 = lines("2", "1. Body text of the direction.", page=2) + [
        Line("8 Inserted vide circular dated June 19, 2018.", 9.0, 700, 708, 2)
    ]
    kept, stats = remove_noise(page1 + page2, pages=2)
    assert [line.text for line in kept] == ["1. Body text of the direction."]
    assert stats["footnote_lines"] == 1
    assert stats["toc_lines"] == 3


def test_numbered_sub_heading_starts_a_new_section():
    paras = build_paragraphs(
        lines(
            "9. The policy shall cover redressal of grievances.",
            "B.2 Review by the Audit Committee of the Board",
            "10. The Audit Committee shall review compliance.",
        )
    )
    assert paras[0].text == "9. The policy shall cover redressal of grievances."
    assert paras[1].kind == "heading"
    assert paras[2].heading == "B.2 Review by the Audit Committee of the Board"


def test_sentence_starting_with_a_is_not_a_heading():
    paras = build_paragraphs(lines("1. Banks shall report.", "A Bank shall also inform the RBI."))
    assert all(p.kind == "text" for p in paras)


def test_non_latin_letterhead_detected():
    assert mostly_non_latin("भारतीय रिज़र्व बैंक")
    assert not mostly_non_latin("RESERVE BANK OF INDIA")
