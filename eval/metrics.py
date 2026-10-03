"""Retrieval and answer metrics.

Retrieval (answerable questions only). For each retrieved chunk we know which evidence quotes
it holds (see eval.testset.is_relevant). A question with two evidence quotes needs both.

    recall@k   share of the question's evidence found in the top k chunks
    hit@k      1 if all evidence is found in the top k (the answer can be complete)
    mrr        1 / rank of the first chunk holding any evidence (0 if none in the list)
    ndcg@k     rank-aware score: relevant chunks near the top count more; the ideal ranking
               puts one relevant chunk per evidence quote first

Answers are summarised in eval.run from per-question records.
"""

import math
import statistics
from collections.abc import Sequence

from eval.testset import Evidence, is_relevant, normalise


def coverage(retrieved: Sequence[tuple[str, str]], evidence: list[Evidence]) -> list[set[int]]:
    """For each retrieved (doc_id, text), the indices of the evidence quotes it holds."""
    out = []
    for doc_id, text in retrieved:
        norm = normalise(text)
        out.append({i for i, ev in enumerate(evidence) if is_relevant(doc_id, norm, ev)})
    return out


def retrieval_scores(covered: list[set[int]], n_evidence: int, ks=(1, 3, 5, 10)) -> dict:
    scores = {}
    for k in ks:
        found = set().union(*covered[:k]) if covered[:k] else set()
        scores[f"recall@{k}"] = len(found) / n_evidence
        scores[f"hit@{k}"] = float(len(found) == n_evidence)
    first = next((rank for rank, c in enumerate(covered, 1) if c), None)
    scores["mrr"] = 1 / first if first else 0.0
    k = 5
    dcg = sum(1 / math.log2(rank + 1) for rank, c in enumerate(covered[:k], 1) if c)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_evidence, k) + 1))
    scores[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return scores


def mean_scores(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {key: round(statistics.fmean(r[key] for r in rows), 4) for key in rows[0]}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]
