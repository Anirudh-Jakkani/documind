"""Cross-encoder reranking.

The first-stage search (BM25 / dense / hybrid) is fast but scores the query and each chunk
separately. A cross-encoder reads the query and a chunk *together*, so it can judge whether
the chunk actually answers the question, at the cost of one model pass per candidate.
We rerank only the top candidates (TOP_K_CANDIDATES, default 30).
"""

from documind.chunking.chunkers import Chunk

RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M parameters: fast
    "bge": "BAAI/bge-reranker-base",  # 278M parameters: stronger, slower
}


class Reranker:
    def __init__(self, name_or_model: str):
        from sentence_transformers import CrossEncoder

        self.model_name = RERANKERS.get(name_or_model, name_or_model)
        self.model = CrossEncoder(self.model_name, device="cpu", max_length=512)

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[tuple[int, float]]:
        """Return (index into `chunks`, score) for the k best chunks."""
        if not chunks:
            return []
        scores = self.model.predict(
            [(query, c.index_text) for c in chunks], batch_size=16, show_progress_bar=False
        )
        ranked = sorted(enumerate(scores), key=lambda pair: -pair[1])
        return [(i, float(s)) for i, s in ranked[:k]]
