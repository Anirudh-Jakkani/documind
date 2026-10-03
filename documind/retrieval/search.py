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
            raise FileNotFoundError(
                f"No index for '{self.strategy}'. Run: "
                f"uv run python -m documind.index --strategy {self.strategy}"
            )
        self.chunks = load_chunks(paths["chunks"])
        self.bm25 = BM25Index.load(paths["bm25"])
        self.qdrant = QdrantClient(path=str(paths["qdrant"]))
        self.collection = collection_name(self.strategy, self.settings.embedding_model)
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings.embedding_model)
        return self._embedder

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        vector = self.embedder.embed_query(query).tolist()
        points = self.qdrant.query_points(self.collection, query=vector, limit=k).points
        return [(point.id, point.score) for point in points]

    def search(self, query: str, mode: str | None = None, k: int | None = None) -> list[Hit]:
        mode = mode or self.settings.retrieval_mode
        k = k or self.settings.top_k_final
        n = max(k, self.settings.top_k_candidates)

        if mode == "dense":
            return [
                Hit(self.chunks[i], s, dense_rank=r)
                for r, (i, s) in enumerate(self.dense_search(query, k), 1)
            ]
        if mode == "bm25":
            return [
                Hit(self.chunks[i], s, bm25_rank=r)
                for r, (i, s) in enumerate(self.bm25.search(query, k), 1)
            ]
        if mode != "hybrid":
            raise ValueError(f"unknown mode {mode!r}")

        dense = [i for i, _ in self.dense_search(query, n)]
        keyword = [i for i, _ in self.bm25.search(query, n)]
        dense_rank = {i: r for r, i in enumerate(dense, 1)}
        bm25_rank = {i: r for r, i in enumerate(keyword, 1)}
        return [
            Hit(self.chunks[i], score, dense_rank.get(i), bm25_rank.get(i))
            for i, score in reciprocal_rank_fusion([dense, keyword])[:k]
        ]

    def close(self) -> None:
        self.qdrant.close()
