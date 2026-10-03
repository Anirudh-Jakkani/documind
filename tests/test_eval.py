import httpx
import pytest

from documind.generation.llm import is_daily_limit
from eval.judge import build_prompt, parse_verdict
from eval.metrics import coverage, percentile, retrieval_scores
from eval.testset import Evidence

QUOTE_A = "any request for closure of a credit card shall be honoured within seven working days"
QUOTE_B = "a penalty of rs 500 per calendar day of delay payable to the cardholder"


def test_retrieval_scores_single_evidence_at_rank_two():
    scores = retrieval_scores([set(), {0}, set()], n_evidence=1, ks=(1, 3))
    assert scores["recall@1"] == 0 and scores["recall@3"] == 1
    assert scores["hit@3"] == 1 and scores["mrr"] == 0.5
    assert 0 < scores["ndcg@5"] < 1


def test_retrieval_scores_multi_evidence_needs_both():
    scores = retrieval_scores([{0}, set(), set()], n_evidence=2, ks=(3,))
    assert scores["recall@3"] == 0.5 and scores["hit@3"] == 0
    perfect = retrieval_scores([{0}, {1}], n_evidence=2, ks=(3,))
    assert perfect["ndcg@5"] == pytest.approx(1.0)


def test_coverage_maps_chunks_to_evidence():
    evidence = [Evidence("rbi-1", QUOTE_A), Evidence("rbi-1", QUOTE_B)]
    retrieved = [("rbi-1", "Chapter II. " + QUOTE_A.upper() + "."), ("rbi-2", "unrelated text")]
    assert coverage(retrieved, evidence) == [{0}, set()]


def test_percentile():
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    assert percentile([], 95) == 0


def test_parse_verdict_accepts_valid_and_rejects_unknown_labels():
    verdict = parse_verdict(
        '{"correctness": "Correct", "faithfulness": "partially supported", "reason": "ok"}'
    )
    assert verdict == {
        "correctness": "correct",
        "faithfulness": "partially_supported",
        "reason": "ok",
    }
    with pytest.raises(ValueError):
        parse_verdict('{"correctness": "great", "faithfulness": "supported"}')


def test_judge_prompt_lists_cited_sources():
    prompt = build_prompt("Q?", "Ref.", "Ans [1].", [{"n": 1, "text": "Source text."}])
    assert (
        "[1] Source text." in prompt and "Reference answer (checked by reviewers): Ref." in prompt
    )
    assert "(no sources cited)" in build_prompt("Q?", "Ref.", "Ans.", [])


def test_daily_limit_detection():
    groq = httpx.Response(429, text='{"error": {"message": "Rate limit ... tokens per day (TPD)"}}')
    minute = httpx.Response(429, text='{"error": {"message": "tokens per minute (TPM)"}}')
    gemini = httpx.Response(429, text='"quotaId": "GenerateRequestsPerDayPerProjectPerModel"')
    assert is_daily_limit(groq) and is_daily_limit(gemini)
    assert not is_daily_limit(minute)
