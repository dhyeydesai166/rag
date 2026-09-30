import json
import time
from pathlib import Path

import pytest
from eval_metrics import (
    aggregate,
    answer_ok,
    expected_keys,
    format_report,
    retrieval_recall,
    retrieved_keys,
)
from eval_set import CASES

from adpater.database_adapter import DatabaseAdapter
from adpater.embedding_adapter import EmbeddingAdapter
from adpater.generation_adapter import GenerationAdapter
from adpater.rerank_adapter import RerankerAdapter
from rag.config import ROUTE_MODEL
from rag.generate import generate
from rag.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[1]


def test_eval_set_covers_known_chunks():
    assert len(CASES) >= 8
    stored = {row["id"] for row in json.loads((ROOT / "chunks.json").read_text())}
    for case in CASES:
        assert case["chunks"]
        assert case["must_contain"]
        assert "must_not_contain" in case
        assert set(case["chunks"]) <= stored


def test_retrieval_recall_counts_overlap_only():
    assert retrieval_recall({"a", "b"}, {"b", "c", "d"}) == 0.5
    assert retrieval_recall({"a"}, set()) == 0.0


def test_compare_keys_use_version_and_heading():
    chunk_id = "Time and Usage Policy|2.0|3. Video Game Time > 3.1 Daily Allowance"
    assert expected_keys("lookup", [chunk_id]) == {chunk_id}
    assert expected_keys("compare", [chunk_id]) == {
        "2.0|3. Video Game Time > 3.1 Daily Allowance"
    }
    hits = [
        {
            "heading_path": "3. Video Game Time > 3.1 Daily Allowance",
            "current": {"version": "2.0"},
            "previous": {"version": "1.0"},
        }
    ]
    assert retrieved_keys("compare", hits) == {
        "2.0|3. Video Game Time > 3.1 Daily Allowance",
        "1.0|3. Video Game Time > 3.1 Daily Allowance",
    }


def test_answer_ok_requires_phrases_and_rejects_banned_ones():
    assert answer_ok("Issued 500,000 tokens.", ["500,000"], ["two million"])
    assert not answer_ok("Issued tokens.", ["500,000"], [])
    assert not answer_ok("500,000 tokens over six hours.", ["500,000"], ["six hours"])
    assert answer_ok(
        "Employees get 7 paid days off.",
        [["7 days", "7 paid days"]],
        ["unpaid"],
    )


def test_aggregate_macro_averages_each_question():
    totals = aggregate(
        [
            {"recall": 1.0, "answer_ok": True},
            {"recall": 0.0, "answer_ok": False},
        ]
    )
    assert totals == {"recall": 0.5, "accuracy": 0.5}
    report = format_report(
        [
            {
                "question": "who gets cake?",
                "recall": 1.0,
                "answer_ok": True,
                "latency": 1.25,
            }
        ],
        {"recall": 1.0, "accuracy": 1.0},
    )
    assert "recall=1.000" in report
    assert "accuracy=1.000" in report
    assert "1.250s" in report


@pytest.mark.ollama
def test_fixed_set_retrieval_and_answers(capsys):
    embedder = EmbeddingAdapter()
    router = GenerationAdapter(model=ROUTE_MODEL)
    answerer = GenerationAdapter()
    database = DatabaseAdapter(ROOT / "chroma")
    reranker = RerankerAdapter()
    rows = []
    for case in CASES:
        started = time.perf_counter()
        found = retrieve(
            case["question"],
            embedder,
            router,
            database,
            reranker,
        )
        answer = generate(case["question"], found["kind"], found["hits"], answerer)
        elapsed = time.perf_counter() - started
        expected = expected_keys(found["kind"], case["chunks"])
        retrieved = retrieved_keys(found["kind"], found["hits"])
        recall = retrieval_recall(expected, retrieved)
        matched = answer_ok(answer, case["must_contain"], case["must_not_contain"])
        rows.append(
            {
                "question": case["question"],
                "recall": recall,
                "answer_ok": matched,
                "latency": elapsed,
                "missing_chunks": sorted(expected - retrieved),
                "answer": answer,
            }
        )
    totals = aggregate(rows)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "result.json").write_text(
        json.dumps({"totals": totals, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    report = format_report(rows, totals)
    with capsys.disabled():
        print(report)
    misses = [
        row["question"] for row in rows if row["recall"] < 1 or not row["answer_ok"]
    ]
    assert misses == [], report
