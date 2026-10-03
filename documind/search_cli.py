"""Search the index from the command line.

uv run python -m documind.search_cli "What is the LRS limit per year?"
uv run python -m documind.search_cli "credit card closure" --mode bm25 --strategy fixed -k 3
"""

import argparse
import sys
import textwrap

from documind.chunking.chunkers import STRATEGIES
from documind.retrieval.search import Retriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Search DocuMind")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    retriever = Retriever(args.strategy)
    hits = retriever.search(args.query, args.mode, args.k)
    print(f'\n"{args.query}"  ({args.mode}, {retriever.strategy} chunks)\n')
    for rank, hit in enumerate(hits, 1):
        ranks = f"dense #{hit.dense_rank or '-'}  bm25 #{hit.bm25_rank or '-'}"
        print(f"{rank}. {hit.chunk.citation()}")
        print(f"   score {hit.score:.4f}   {ranks}   {hit.chunk.n_tokens} tokens")
        print(textwrap.indent(textwrap.fill(hit.chunk.text[:400] + " ...", 96), "   "))
        print()
    retriever.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
