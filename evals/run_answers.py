"""Answer eval over N runs. Prints misses and inventions; does not fail unless
--fail-on-problems is set.
"""

import argparse
import json
import time
from pathlib import Path

import ollama

from adapter.database_adapter import open_active_index
from adapter.embedding_adapter import EmbeddingAdapter
from adapter.generation_adapter import GenerationAdapter
from adapter.rerank_adapter import make_reranker
from evals.cases import CASES
from evals.checks import check_answer, counterparts_by_id, passages_by_id
from evals.pacing import pause_between_questions
from rag.config import ANSWER_EVAL_RUNS, OLLAMA_HOST
from rag.generate import generate, missing_side_lines
from rag.manifest import load_manifest
from rag.provenance import provenance
from rag.retrieve import _check_models, retrieve
from rag.route import embed_examples

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def score_case(case: dict, found: dict, model) -> dict:
    started = time.perf_counter()
    generated = generate(case["question"], found, model)
    answer = generated.answer
    lines = []
    if answer.status != "not_in_sources":
        cited_ids = {claim.chunk_id for claim in answer.claims}
        lines = missing_side_lines(found.get("route") or {}, found["hits"], cited_ids)
    checked = check_answer(
        answer,
        passages_by_id(found["kind"], found["hits"]),
        found["kind"],
        case,
        counterparts_by_id(found["kind"], found["hits"]),
        display_lines=lines,
        reranked=bool(found.get("reranked")),
    )
    return {
        "id": case["id"],
        "question": case["question"],
        "route": found["kind"],
        "route_ok": found["kind"] == case["route"],
        "status": answer.status,
        "answer": answer.model_dump(),
        "text": generated.text,
        "retrieved_ids": list(passages_by_id(found["kind"], found["hits"])),
        "misses": checked.misses,
        "inventions": checked.inventions,
        "skipped": bool(case.get("needs_rerank")) and not bool(found.get("reranked")),
        "latency": time.perf_counter() - started,
    }


def totals(rows: list[dict]) -> dict:
    return {
        "misses": sum(len(row["misses"]) for row in rows),
        "inventions": sum(len(row["inventions"]) for row in rows),
        "clean_cases": sum(
            1
            for row in rows
            if not row.get("skipped") and not row["misses"] and not row["inventions"]
        ),
        "route_ok": sum(1 for row in rows if row["route_ok"]),
        "skipped": sum(1 for row in rows if row.get("skipped")),
    }


def format_totals(run: int, summary: dict) -> str:
    return (
        f"{run:>3}  {summary['misses']:>7}  {summary['inventions']:>11}  "
        f"{summary['clean_cases']:>11}  {summary['route_ok']:>8}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run_answers")
    parser.add_argument("--runs", type=int, default=ANSWER_EVAL_RUNS)
    parser.add_argument("--db", default="chroma")
    parser.add_argument("--fail-on-problems", action="store_true")
    args = parser.parse_args(argv)
    _check_models()
    client = ollama.Client(host=OLLAMA_HOST)
    embedder = EmbeddingAdapter()
    database = open_active_index(args.db)
    reranker = make_reranker()
    model = GenerationAdapter()
    examples = embed_examples(embedder)
    manifest = load_manifest(args.db)
    build_id = manifest.get("active_build_id") or ""
    build = {**manifest.get("builds", {}).get(build_id, {}), "build_id": build_id}
    runs = []
    print("run  misses  inventions  clean_cases  route_ok")
    for number in range(1, args.runs + 1):
        rows = []
        for index, case in enumerate(CASES):
            rows.append(
                score_case(
                    case,
                    retrieve(case["question"], embedder, database, reranker, examples),
                    model,
                )
            )
            if index < len(CASES) - 1 or number < args.runs:
                pause_between_questions(reranker)
        summary = totals(rows)
        runs.append({"run": number, "totals": summary, "cases": rows})
        print(format_totals(number, summary))
    payload = {
        "provenance": provenance(client, build),
        "runs_requested": args.runs,
        "runs": runs,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answer_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    problems = sum(
        run["totals"]["misses"] + run["totals"]["inventions"] for run in runs
    )
    if args.fail_on_problems and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
