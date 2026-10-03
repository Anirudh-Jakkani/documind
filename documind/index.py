"""Build the chunk, BM25 and vector indexes for one or all chunking strategies.

uv run python -m documind.index --strategy section
uv run python -m documind.index --strategy all
"""

import argparse
import json
import statistics
import sys
import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from documind.chunking.chunkers import STRATEGIES, chunk_document, load_parsed, save_chunks
from documind.config import get_settings
from documind.retrieval.bm25 import BM25Index
from documind.retrieval.embedder import Embedder, TokenCounter
from documind.retrieval.search import collection_name, index_paths

UPSERT_BATCH = 256


def remove_stale_points(client: QdrantClient, name: str, n_chunks: int) -> None:
    stale, offset = [], None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset, with_payload=False)
        stale.extend(p.id for p in points if p.id >= n_chunks)
        if offset is None:
            break
    if stale:
        client.delete(name, points_selector=PointIdsList(points=stale))
        print(f"  removed {len(stale)} stale vector(s) left from an earlier index")
    count = client.count(name).count
    if count != n_chunks:
        raise RuntimeError(f"{name}: {count} vectors for {n_chunks} chunks")


def build(strategy: str, docs: list[dict], count: TokenCounter, embedder: Embedder) -> dict:
    settings = get_settings()
    paths = index_paths(settings, strategy)
    started = time.perf_counter()

    chunks = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                strategy,
                count,
                settings.chunk_size_tokens,
                settings.chunk_overlap_tokens,
                settings.section_min_tokens,
            )
        )
    save_chunks(chunks, paths["chunks"])
    texts = [c.index_text for c in chunks]
    BM25Index().build(texts).save(paths["bm25"])

    print(f"[{strategy}] embedding {len(chunks):,} chunks with {settings.embedding_model} ...")
    vectors = embedder.embed_documents(texts)

    client = QdrantClient(path=str(paths["qdrant"]))
    name = collection_name(strategy, settings.embedding_model)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=VectorParams(size=embedder.dimension, distance=Distance.COSINE)
    )
    for start in range(0, len(chunks), UPSERT_BATCH):
        client.upsert(
            name,
            points=[
                PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"chunk_id": chunks[i].chunk_id, "doc_id": chunks[i].doc_id},
                )
                for i in range(start, min(start + UPSERT_BATCH, len(chunks)))
            ],
        )
    # On Windows, deleting a local collection can leave old points behind, so remove any
    # point that doesn't belong to the new chunk list and check the counts match.
    remove_stale_points(client, name, len(chunks))
    client.close()

    sizes = sorted(c.n_tokens for c in chunks)
    stats = {
        "strategy": strategy,
        "embedding_model": settings.embedding_model,
        "chunks": len(chunks),
        "tokens_median": int(statistics.median(sizes)),
        "tokens_p95": sizes[int(len(sizes) * 0.95)],
        "tokens_max": sizes[-1],
        "over_512": sum(s > 512 for s in sizes),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (paths["dir"] / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DocuMind indexes")
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="section")
    args = parser.parse_args()

    settings = get_settings()
    docs = load_parsed(settings.data_dir / "processed")
    if not docs:
        print("No parsed documents. Run: uv run python -m documind.ingest.parse")
        return 1
    count = TokenCounter(settings.embedding_model)
    embedder = Embedder(settings.embedding_model)

    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    results = [build(s, docs, count, embedder) for s in strategies]

    columns = ("chunks", "median", "p95", "max", ">512", "time")
    print(f"\n{'strategy':<10} " + " ".join(f"{c:>7}" for c in columns))
    for r in results:
        print(
            f"{r['strategy']:<10} {r['chunks']:>7,} {r['tokens_median']:>7} {r['tokens_p95']:>5} "
            f"{r['tokens_max']:>5} {r['over_512']:>5} {r['seconds']:>6}s"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
