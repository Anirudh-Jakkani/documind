"""Local embedding model and tokenizer (sentence-transformers)."""

from functools import lru_cache

import numpy as np

# Models trained with an instruction for queries (and, for e5, for passages).
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
}
PASSAGE_PREFIX = {"intfloat/e5-base-v2": "passage: "}


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


class TokenCounter:
    """Counts tokens with the embedding model's own tokenizer, so chunks fit the model."""

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @lru_cache(maxsize=200_000)  # noqa: B019 - one long-lived instance per run
    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        prefix = PASSAGE_PREFIX.get(self.model_name, "")
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def embed_query(self, text: str) -> np.ndarray:
        prefix = QUERY_PREFIX.get(self.model_name, "")
        return self.model.encode(prefix + text, normalize_embeddings=True, convert_to_numpy=True)
