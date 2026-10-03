"""Ask a question from the command line.

uv run python -m documind.ask "What is the yearly LRS limit?"
uv run python -m documind.ask "..." --mode dense -k 8 --show-sources
"""

import argparse
import sys
import textwrap

from documind.generation.answer import Answerer


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask DocuMind a question")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default=None)
    parser.add_argument("-k", type=int, default=None, help="number of chunks to give the model")
    parser.add_argument("--show-sources", action="store_true", help="print the source text")
    args = parser.parse_args()

    answerer = Answerer()
    result = answerer.answer(args.question, args.mode, args.k)
    answerer.close()

    print(f"\nQ: {result.question}\n")
    print(textwrap.fill(result.answer, 96))
    if result.found:
        print("\nSources:")
        for source in result.sources:
            print(f"  [{source.n}] {source.citation}")
            if args.show_sources:
                print(textwrap.indent(textwrap.fill(source.text, 92), "      "))
    else:
        print("\n(not found in the documents)")
    if result.invalid_citations:
        print(f"\nWarning: the model cited sources that don't exist: {result.invalid_citations}")
    t, tok = result.timings, result.tokens
    print(
        f"\n{t['total_s']}s (search {t['retrieval_s']}s, LLM {t['llm_s']}s) · "
        f"{tok['prompt']:,} + {tok['completion']:,} tokens"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
