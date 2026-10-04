"""Evaluate DocuMind on eval/dataset.jsonl.

Retrieval only (no LLM calls, free):
    uv run python -m eval.run retrieval                       # every strategy x mode
    uv run python -m eval.run retrieval --strategies section --modes hybrid

Full answers, graded by the judge model (resumes if interrupted):
    uv run python -m eval.run answers --name baseline         # section + hybrid, top 5
    uv run python -m eval.run answers --name dense-k8 --mode dense -k 8

Results go to eval/results/: retrieval.csv, answers_<name>.jsonl (one line per question)
and summary_<name>.json. `uv run python -m eval.run report` prints all answer summaries.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict

from documind.chunking.chunkers import STRATEGIES
from documind.config import get_settings
from documind.generation.answer import Answerer
from documind.generation.llm import DailyLimitError, LLMClient, LLMError
from documind.retrieval.embedder import Embedder, model_slug
from documind.retrieval.reranker import RERANKERS, Reranker
from documind.retrieval.rewrite import QueryRewriter
from documind.retrieval.search import Retriever
from eval.judge import judge
from eval.metrics import coverage, mean_scores, percentile, retrieval_scores
from eval.paraphrase import PARAPHRASED_PATH
from eval.testset import DATASET_PATH, EVAL_DIR, Record, load_records

RESULTS_DIR = EVAL_DIR / "results"
MODES = ("dense", "bm25", "hybrid")
DATASETS = {"original": DATASET_PATH, "paraphrased": PARAPHRASED_PATH}


def doc_of(chunk_id: str) -> str:
    return chunk_id.split(":")[0]


# --- Retrieval ----------------------------------------------------------------------------

CONFIG_KEYS = ("dataset", "strategy", "mode", "embedding", "rerank", "rewrite")
DEFAULT_CONFIG = {
    "dataset": "original",
    "embedding": "bge-small-en-v1.5",
    "rerank": "none",
    "rewrite": "no",
}


def load_dataset(name: str) -> list[Record]:
    return load_records(DATASETS[name])


def make_rewriter() -> QueryRewriter:
    RESULTS_DIR.mkdir(exist_ok=True)
    return QueryRewriter(LLMClient(), cache_path=RESULTS_DIR / "rewrite_cache.json")


def evaluate_retrieval(
    records: list[Record], retriever: Retriever, mode: str, k: int, reranker=None, rewriter=None
) -> dict:
    rows, started = [], time.perf_counter()
    for r in records:
        extra = rewriter.rewrite(r.question) if rewriter else None
        hits = retriever.search(r.question, mode, k, reranker=reranker, extra_queries=extra)
        covered = coverage([(h.chunk.doc_id, h.chunk.text) for h in hits], r.evidence)
        rows.append(retrieval_scores(covered, len(r.evidence)))
    summary = mean_scores(rows)
    summary["ms_per_query"] = round(1000 * (time.perf_counter() - started) / len(records), 1)
    return summary


RETRIEVAL_PATH = RESULTS_DIR / "retrieval.csv"


def load_retrieval() -> list[dict]:
    if not RETRIEVAL_PATH.exists():
        return []
    with RETRIEVAL_PATH.open(encoding="utf-8", newline="") as f:
        return [{**DEFAULT_CONFIG, **row} for row in csv.DictReader(f)]


def save_retrieval(new_rows: list[dict]) -> None:
    """Merge with earlier runs: a new result replaces an old one with the same configuration."""
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = {tuple(str(r[c]) for c in CONFIG_KEYS): r for r in load_retrieval()}
    for row in new_rows:
        rows[tuple(str(row[c]) for c in CONFIG_KEYS)] = row
    fields = [*CONFIG_KEYS, *[c for c in new_rows[0] if c not in CONFIG_KEYS]]
    with RETRIEVAL_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows.values())


def run_retrieval(args) -> int:
    settings = get_settings()
    embedder = Embedder(settings.embedding_model)
    reranker = Reranker(args.rerank) if args.rerank != "none" else None
    rewriter = make_rewriter() if args.rewrite else None
    results = []
    for dataset in args.datasets:
        records = [r for r in load_dataset(dataset) if r.kind != "unanswerable"]
        for strategy in args.strategies:
            retriever = Retriever(strategy, embedder=embedder)
            for mode in args.modes:
                config = {
                    "dataset": dataset,
                    "strategy": strategy,
                    "mode": mode,
                    "embedding": model_slug(settings.embedding_model),
                    "rerank": args.rerank,
                    "rewrite": "yes" if args.rewrite else "no",
                }
                scores = evaluate_retrieval(records, retriever, mode, args.k, reranker, rewriter)
                results.append({**config, **scores})
                save_retrieval([results[-1]])  # saved per configuration: a stop loses one at most
                print(
                    f"  {' / '.join(config.values())}: recall@5 {scores['recall@5']:.3f}  "
                    f"MRR {scores['mrr']:.3f}"
                )
            retriever.close()

    print_retrieval_table(load_retrieval())
    return 0


def print_retrieval_table(rows: list[dict]) -> None:
    cols = ["recall@1", "recall@5", "recall@10", "mrr", "ms_per_query"]
    print("\n" + "".join(f"{c:<12}" for c in CONFIG_KEYS) + "".join(f"{c:>11}" for c in cols))
    for r in sorted(rows, key=lambda r: (r["dataset"], -float(r["recall@5"]))):
        print(
            "".join(f"{str(r[c])[:11]:<12}" for c in CONFIG_KEYS)
            + "".join(f"{float(r[c]):>11.3f}" for c in cols)
        )


# --- Answers ------------------------------------------------------------------------------


def answer_one(record: Record, answerer: Answerer, judge_llm: LLMClient, mode, k) -> dict:
    result = answerer.answer(record.question, mode, k)
    retrieved = [(doc_of(s.chunk_id), s.text) for s in result.retrieved]
    row = {
        "id": record.id,
        "kind": record.kind,
        "question": record.question,
        "reference": record.answer,
        "answer": result.answer,
        "found": result.found,
        "cited": [s.n for s in result.sources],
        "invalid_citations": result.invalid_citations,
        "timings": result.timings,
        "tokens": result.tokens,
        "model": answerer.llm.model,
        "retrieved": [
            {"n": s.n, "chunk_id": s.chunk_id, "citation": s.citation} for s in result.retrieved
        ],
    }
    if record.kind == "unanswerable":
        row["refusal_correct"] = not result.found
        return row

    covered = coverage(retrieved, record.evidence)
    row["retrieval"] = retrieval_scores(covered, len(record.evidence), ks=(1, 3, 5))
    cited_cover = set().union(*(covered[n - 1] for n in row["cited"])) if row["cited"] else set()
    row["cites_evidence"] = bool(cited_cover)
    row["cites_all_evidence"] = len(cited_cover) == len(record.evidence)
    if result.found:
        sources = [asdict(s) for s in result.sources]
        verdict, tokens = judge(judge_llm, record.question, record.answer, result.answer, sources)
        row.update(verdict)
        row["judge_tokens"] = tokens
    return row


def summarise(rows: list[dict], config: dict) -> dict:
    answerable = [r for r in rows if r["kind"] != "unanswerable"]
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    answered = [r for r in answerable if r["found"]]
    judged = [r for r in answered if "correctness" in r]

    def share(items, test):
        return round(sum(1 for i in items if test(i)) / len(items), 4) if items else None

    correct_score = [
        {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[r["correctness"]] for r in judged
    ]
    total_s = [r["timings"]["total_s"] for r in rows]
    return {
        **config,
        "questions": len(rows),
        "answerable": {
            "n": len(answerable),
            "answered": share(answerable, lambda r: r["found"]),
            "false_refusal": share(answerable, lambda r: not r["found"]),
            "correct": share(answerable, lambda r: r.get("correctness") == "correct"),
            "correct_or_partial": share(
                answerable, lambda r: r.get("correctness") in ("correct", "partial")
            ),
            "correctness_score_when_answered": round(statistics.fmean(correct_score), 4)
            if correct_score
            else None,
            "faithful_when_answered": share(judged, lambda r: r["faithfulness"] == "supported"),
            "cites_evidence_when_answered": share(answered, lambda r: r["cites_evidence"]),
            "invalid_citation_rate": share(answerable, lambda r: bool(r["invalid_citations"])),
            "retrieval": mean_scores([r["retrieval"] for r in answerable]),
        },
        "unanswerable": {
            "n": len(unanswerable),
            "correct_refusal": share(unanswerable, lambda r: r["refusal_correct"]),
        },
        "multi_part_correct": share(
            [r for r in answerable if r["kind"] == "multi"],
            lambda r: r.get("correctness") == "correct",
        ),
        "latency_s": {
            "p50": round(statistics.median(total_s), 2),
            "p95": round(percentile(total_s, 95), 2),
        },
        "tokens_per_question": {
            "prompt": round(statistics.fmean(r["tokens"]["prompt"] for r in rows)),
            "completion": round(statistics.fmean(r["tokens"]["completion"] for r in rows)),
        },
    }


def run_answers(args) -> int:
    settings = get_settings()
    records = load_dataset(args.dataset)
    if args.limit:
        records = records[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"answers_{args.name}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    todo = [r for r in records if r.id not in done]
    strategy = args.strategy or settings.chunk_strategy
    mode = args.mode or settings.retrieval_mode
    k = args.k or settings.top_k_final
    print(f"{len(done)} already done, {len(todo)} to go ({strategy}, {mode}, top {k})")

    answerer = Answerer(
        Retriever(strategy),
        LLMClient(),
        reranker=Reranker(args.rerank) if args.rerank != "none" else None,
        rewriter=make_rewriter() if args.rewrite else None,
    )
    judge_llm = LLMClient(role="judge")
    config = {
        "name": args.name,
        "dataset": args.dataset,
        "strategy": strategy,
        "mode": mode,
        "k": k,
        "embedding": model_slug(settings.embedding_model),
        "rerank": args.rerank,
        "rewrite": "yes" if args.rewrite else "no",
        "answer_model": f"{answerer.llm.provider}:{answerer.llm.model}",
        "judge_model": f"{judge_llm.provider}:{judge_llm.model}",
    }
    stopped = None
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for record in todo:
                try:
                    row = answer_one(record, answerer, judge_llm, mode, k)
                except DailyLimitError as error:
                    stopped = str(error)
                    break
                except (LLMError, ValueError) as error:
                    print(f"  {record.id}: {error} (skipped, will retry on the next run)")
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[record.id] = row
                verdict = row.get("correctness") or ("refused" if not row["found"] else "answered")
                print(f"  [{len(done)}/{len(records)}] {record.id} {verdict}")
    finally:
        answerer.close()
        judge_llm.close()

    rows = [done[r.id] for r in records if r.id in done]
    summary = summarise(rows, config)
    (RESULTS_DIR / f"summary_{args.name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print_summary(summary)
    if stopped:
        print(f"\nStopped early: {stopped}\nRun the same command later to continue.")
        return 2
    if len(rows) < len(records):
        print(f"\n{len(records) - len(rows)} questions failed; run the same command to retry.")
    return 0


def print_summary(s: dict) -> None:
    a, u = s["answerable"], s["unanswerable"]
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    print(
        f"\n=== {s['name']}: {s['strategy']} / {s['mode']} / top {s['k']} "
        f"· {s['answer_model']} · judge {s['judge_model']}"
    )
    print(
        f"Answerable ({a['n']}):  correct {pct(a['correct'])}  "
        f"correct-or-partial {pct(a['correct_or_partial'])}  "
        f"false refusals {pct(a['false_refusal'])}"
    )
    print(
        f"  when answered: faithful {pct(a['faithful_when_answered'])}  "
        f"cites the evidence {pct(a['cites_evidence_when_answered'])}  "
        f"invalid citations {pct(a['invalid_citation_rate'])}"
    )
    print(
        f"  retrieval: recall@5 {a['retrieval'].get('recall@5')}  MRR {a['retrieval'].get('mrr')}"
    )
    print(f"Multi-part correct: {pct(s['multi_part_correct'])}")
    print(f"Unanswerable ({u['n']}): correctly refused {pct(u['correct_refusal'])}")
    print(
        f"Latency p50 {s['latency_s']['p50']}s, p95 {s['latency_s']['p95']}s · "
        f"tokens/question {s['tokens_per_question']}"
    )


def run_report(_args) -> int:
    for path in sorted(RESULTS_DIR.glob("summary_*.json")):
        print_summary(json.loads(path.read_text(encoding="utf-8")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DocuMind")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("retrieval", help="retrieval metrics, no LLM calls")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--datasets", nargs="+", default=["original"], choices=list(DATASETS))
    p.add_argument("--rerank", default="none", choices=["none", *RERANKERS])
    p.add_argument("--rewrite", action="store_true", help="LLM query rewriting (cached)")
    p.set_defaults(func=run_retrieval)

    p = sub.add_parser("answers", help="answer every question and grade it")
    p.add_argument("--name", required=True)
    p.add_argument("--strategy", choices=STRATEGIES)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("-k", type=int)
    p.add_argument("--limit", type=int, help="only the first N questions (for a quick test)")
    p.add_argument("--dataset", default="original", choices=list(DATASETS))
    p.add_argument("--rerank", default="none", choices=["none", *RERANKERS])
    p.add_argument("--rewrite", action="store_true", help="LLM query rewriting (cached)")
    p.set_defaults(func=run_answers)

    p = sub.add_parser("report", help="print saved answer summaries")
    p.set_defaults(func=run_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
