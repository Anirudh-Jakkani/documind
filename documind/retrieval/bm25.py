"""A small, readable BM25 keyword index.

Written by hand rather than imported so every part of the score can be explained:

    score(q, d) = sum over query terms t of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""

import gzip
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

# Keeps numbers like "2,50,000", "35A" and "5.2" together, which matter in regulations.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,/][0-9]+)*[a-z]?")
STOPWORDS = frozenset(
    """a an and are as at be been by can for from has have if in into is it its may of on or
    shall such that the their there these this those to under was were which will with""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        # Light stemming: "cards" -> "card", but keep "business", "process".
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_len = 0.0

    def build(self, texts: list[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                postings[term].append((doc_id, tf))
        n = len(texts)
        self.postings = dict(postings)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        self.avg_len = sum(self.doc_len) / n if n else 0.0
        return self

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda item: -item[1])[:k]

    def save(self, path: Path) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        index = cls()
        with gzip.open(path, "rb") as f:
            index.__dict__.update(pickle.load(f))
        return index
