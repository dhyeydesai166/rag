"""Deterministic retrieval gate. Exit 1 when gold is missing from the fused list
or the route is wrong. The final top 3 is a gate only when Cohere actually ranked.
"""

import json
import sys
import time
from pathlib import Path

from adapter.database_adapter import open_active_index
from adapter.embedding_adapter import EmbeddingAdapter
from adapter.rerank_adapter import make_reranker
from evals.cases import CASES
from evals.checks import retrieval_hit, retrieved_ids
from evals.pacing import pause_between_questions
from rag.manifest import load_manifest
from rag.provenance import provenance
from rag.retrieve import _check_models, retrieve
from rag.route import embed_examples

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def passes_gate(row: dict) -> bool:
    """Fused shortlist and route always count. Top 3 counts only with Cohere."""
    if not row["route_ok"]:
        return False
    if not row["gold_chunks"]:
        return True
    if not row["gold_in_fused"]:
        return False
    if row["reranked"] and not row["gold_in_top"]:
        return False
    return True


def score_case(case: dict, found: dict) -> dict:
    gold = case["gold_chunks"]
    fused = set(found.get("fused_ids") or [])
    final = set(retrieved_ids(found["kind"], found["hits"]))
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_route": case["route"],
        "route": found["kind"],
        "route_ok": found["kind"] == case["route"],
        "gold_chunks": gold,
        "gold_in_fused": retrieval_hit(gold, fused) if gold else True,
        "gold_in_top": retrieval_hit(gold, final) if gold else True,
        "reranked": bool(found.get("reranked")),
        "fused_ids": list(found.get("fused_ids") or []),
        "final_ids": sorted(final),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    db_path = argv[0] if argv else "chroma"
    _check_models()
    import ollama

    from rag.config import OLLAMA_HOST

    client = ollama.Client(host=OLLAMA_HOST)
    embedder = EmbeddingAdapter()
    database = open_active_index(db_path)
    reranker = make_reranker()
    examples = embed_examples(embedder)
    rows = []
    for index, case in enumerate(CASES):
        started = time.perf_counter()
        found = retrieve(case["question"], embedder, database, reranker, examples)
        row = score_case(case, found)
        row["latency"] = time.perf_counter() - started
        rows.append(row)
        if index < len(CASES) - 1:
            pause_between_questions(reranker)
    manifest = load_manifest(db_path)
    build_id = manifest.get("active_build_id") or ""
    build = manifest.get("builds", {}).get(build_id, {})
    build = {**build, "build_id": build_id}
    payload = {"provenance": provenance(client, build), "cases": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_eval.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    failed = [row["id"] for row in rows if not passes_gate(row)]
    for row in rows:
        mark = "ok" if passes_gate(row) else "FAIL"
        rerank = "reranked" if row["reranked"] else "fused-only"
        print(
            f"{mark:4} {row['id']:28} route={row['route']:7} "
            f"fused={row['gold_in_fused']} top={row['gold_in_top']} {rerank}"
        )
    if failed:
        print(f"retrieval gate failed: {', '.join(failed)}")
        return 1
    print(f"retrieval gate passed ({len(rows)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
