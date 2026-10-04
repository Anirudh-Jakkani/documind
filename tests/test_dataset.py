"""The committed test sets must stay consistent: these run in CI without the index."""

from eval.paraphrase import PARAPHRASED_PATH
from eval.testset import DATASET_PATH, KINDS, load_records


def test_test_set_is_consistent():
    records = load_records(DATASET_PATH)
    assert len(records) == 128
    assert len({r.id for r in records}) == len(records)
    for r in records:
        assert r.kind in KINDS and r.question.strip()
        assert r.status in ("approved", "edited")
        if r.kind == "unanswerable":
            assert not r.evidence
        else:
            assert r.answer.strip() and r.evidence
            assert all(
                e.doc_id.startswith("rbi-") and len(e.quote.split()) >= 8 for e in r.evidence
            )
        if r.kind == "multi":
            assert len(r.evidence) == 2


def test_paraphrased_set_matches_the_original():
    original = {r.id: r for r in load_records(DATASET_PATH)}
    paraphrased = load_records(PARAPHRASED_PATH)
    assert {r.id for r in paraphrased} == set(original)
    for r in paraphrased:
        assert r.evidence == original[r.id].evidence and r.answer == original[r.id].answer
