"""Dense, keyword (BM25) and hybrid search over one chunking strategy's index."""

from dataclasses import dataclass
from pathlib import Path

from documind.chunking.chunkers import Chunk, load_chunks
from documind.config import Settings, get_settings
from documind.retrieval.bm25 import BM25Index
from documind.retrieval.embedder import Embedder, model_slug

RRF_K = 60  # standard constant from the Reciprocal Rank Fusion paper (Cormack et al., 2009)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    first_stage_rank: int | None = None  # rank before reranking


def index_paths(settings: Settings, strategy: str) -> dict[str, Path]:
    base = settings.index_dir / strategy
    return {
        "dir": base,
        "chunks": base / "chunks.jsonl",
        "bm25": base / "bm25.pkl.gz",
        "qdrant": settings.index_dir / "qdrant",
    }


def collection_name(strategy: str, embedding_model: str) -> str:
    return f"{strategy}__{model_slug(embedding_model)}"


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Combine ranked lists: each item scores sum(1 / (k + rank)) over the lists it appears in.
    Uses ranks only, so dense similarities and BM25 scores never need to be put on one scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


class Retriever:
    def __init__(
        self,
        strategy: str | None = None,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ):
        from qdrant_client import QdrantClient

        self.settings = settings or get_settings()
        self.strategy = strategy or self.settings.chunk_strategy
        paths = index_paths(self.settings, self.strategy)
        if not paths["chunks"].exists():
            checked = ", ".join(
                f"{p} ({'found' if p.exists() else 'missing'})"
                for p in (self.settings.index_dir, paths["dir"], paths["chunks"])
            )
            raise FileNotFoundError(
                f"No index for '{self.strategy}'. Checked: {checked}. Build one with: "
                f"uv run python -m documind.index --strategy {self.strategy}"
            )
        self.chunks = load_chunks(paths["chunks"])
        self.bm25 = BM25Index.load(paths["bm25"])
        self.qdrant = QdrantClient(path=str(paths["qdrant"]))
        self.collection = collection_name(self.strategy, self.settings.embedding_model)
        self._embedder = embedder
        self._check_index()

    def _check_index(self) -> None:
        """Vectors and chunks must match one to one, or search returns the wrong text."""
        rebuild = (
            f"Rebuild it: uv run python -m documind.index --strategy {self.strategy} "
            f"(with EMBEDDING_MODEL={self.settings.embedding_model})"
        )
        if not self.qdrant.collection_exists(self.collection):
            raise FileNotFoundError(f"No vector index '{self.collection}'. {rebuild}")
        vectors = self.qdrant.count(self.collection).count
        if vectors != len(self.chunks):
            self.qdrant.close()
            raise RuntimeError(
                f"Index out of date: '{self.collection}' has {vectors} vectors but there are "
                f"{len(self.chunks)} chunks. {rebuild}"
            )

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings.embedding_model)
        return self._embedder

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = self.embedder.embed_query(query).tolist()
        points = self.qdrant.query_points(self.collection, query=vector, limit=k).points
        return [(point.id, point.score) for point in points]

    def first_stage(self, query: str, mode: str, n: int) -> list[tuple[int, float]]:
        """Chunk indices and scores from dense, BM25 or hybrid search."""
        if mode == "dense":
            return self.dense_search(query, n)
        if mode == "bm25":
            return self.bm25.search(query, n)
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")
        dense = [i for i, _ in self.dense_search(query, n)]
        keyword = [i for i, _ in self.bm25.search(query, n)]
        return reciprocal_rank_fusion([dense, keyword])[:n]

    def search(
        self,
        query: str,
        mode: str | None = None,
        k: int | None = None,
        *,
        reranker=None,
        extra_queries: list[str] | None = None,
    ) -> list[Hit]:
        """Top-k chunks for `query`.

        extra_queries: other phrasings of the same question (from query rewriting); each is
            searched and all rankings are fused with reciprocal rank fusion.
        reranker: a Reranker that re-scores the top TOP_K_CANDIDATES with the original query.
        """
        mode = mode or self.settings.retrieval_mode
        k = k or self.settings.top_k_final
        n = max(k, self.settings.top_k_candidates) if (reranker or extra_queries) else k
        if mode == "hybrid":
            n = max(n, self.settings.top_k_candidates)  # fusion needs a deep list from each

        rankings = [self.first_stage(q, mode, n) for q in [query, *(extra_queries or [])]]
        if len(rankings) == 1:
            ranked = rankings[0][:n]
        else:
            ranked = reciprocal_rank_fusion([[i for i, _ in r] for r in rankings])[:n]

        if reranker is not None:
            candidates = [self.chunks[i] for i, _ in ranked]
            first_rank = {i: r for r, (i, _) in enumerate(ranked, 1)}
            return [
                Hit(candidates[j], score, first_stage_rank=first_rank[ranked[j][0]])
                for j, score in reranker.rerank(query, candidates, k)
            ]
        return [Hit(self.chunks[i], score) for i, score in ranked[:k]]

    def close(self) -> None:
        self.qdrant.close()
