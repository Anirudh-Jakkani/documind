from documind.retrieval.bm25 import BM25Index, tokenize
from documind.retrieval.search import reciprocal_rank_fusion


def test_tokenize_keeps_regulatory_numbers_and_stems_plurals():
    assert tokenize("The LRS limit is USD 2,50,000 under Section 35A for cards") == [
        "lrs",
        "limit",
        "usd",
        "2,50,000",
        "section",
        "35a",
        "card",
    ]


def test_bm25_ranks_matching_document_first():
    index = BM25Index().build(
        [
            "Credit card closure requests shall be honoured within seven working days.",
            "Banks shall maintain a cash reserve ratio as prescribed.",
            "Know your customer norms apply to all accounts.",
        ]
    )
    results = index.search("how many days to close a credit card", k=3)
    assert results[0][0] == 0
    assert index.search("nonexistent term xyz", k=3) == []


def test_bm25_rare_terms_weigh_more():
    index = BM25Index().build(["bank bank rare", "bank common", "bank common", "bank common"])
    assert index.idf["rare"] > index.idf["common"] > index.idf["bank"]


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    order = [item for item, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert set(order) == {1, 2, 3, 4}
