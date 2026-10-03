from eval.testset import (
    Evidence,
    Record,
    chunk_contains,
    key_span,
    load_records,
    locate_quote,
    save_records,
)

PASSAGE = (
    "19. Any request for closure of a credit card shall be honoured within seven working days "
    "by the card-issuer, subject to payment of all dues by the cardholder. Failure on the part "
    "of the card-issuer to complete the process of closure within seven working days shall "
    "result in a penalty of ₹500 per day of delay."
)


def test_locate_quote_exact_and_with_typographic_differences():
    assert locate_quote("shall be honoured within seven working days", PASSAGE)
    # Model wrote a straight dash and lost a space: still found.
    assert locate_quote("by the card–issuer, subject to payment of all  dues", PASSAGE)


def test_locate_quote_rejects_invented_text():
    assert locate_quote("shall be honoured within ten calendar days by the bank", PASSAGE) is None
    assert locate_quote("", PASSAGE) is None


def test_chunk_boundary_still_matches_key_span():
    quote = (
        "Failure on the part of the card-issuer to complete the process of closure within "
        "seven working days shall result in a penalty of ₹500 per day of delay."
    )
    evidence = Evidence("rbi-13155", quote)
    # A fixed-size chunk that cut off the first and last few words of the quote.
    cut = PASSAGE[PASSAGE.index("part of the") : PASSAGE.index("per day")]
    assert chunk_contains(cut, evidence)
    assert not chunk_contains("Banks shall maintain a cash reserve ratio.", evidence)
    assert len(key_span(quote).split()) == 12


def test_records_round_trip(tmp_path):
    path = tmp_path / "set.jsonl"
    record = Record("s001", "Q?", "A.", "single", evidence=[Evidence("rbi-1", "quote", "3", 4)])
    save_records([record], path)
    assert load_records(path) == [record]
