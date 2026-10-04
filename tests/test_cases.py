import json
from pathlib import Path

from evals.cases import CASES

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {
    "id",
    "question",
    "route",
    "expect_status",
    "gold_chunks",
    "required_facts",
    "stale_facts",
}


def test_cases_have_the_migrated_set_and_unique_ids():
    assert len(CASES) == 21
    assert len({case["id"] for case in CASES}) == 21
    assert sum(case["route"] == "compare" for case in CASES) == 4


def test_every_case_has_the_required_fields():
    for case in CASES:
        assert FIELDS <= set(case)
        assert case["route"] in {"lookup", "compare"}
        assert case["expect_status"] in {"answered", "not_in_sources", "conflicting"}
        assert "must_not_contain" not in case
        assert "must_contain" not in case


def test_every_gold_id_exists_in_chunks_json():
    stored = {row["id"] for row in json.loads((ROOT / "chunks.json").read_text())}
    for case in CASES:
        assert set(case["gold_chunks"]) <= stored


def test_out_of_scope_case_expects_not_in_sources():
    case = next(item for item in CASES if item["id"] == "travel-out-of-scope")
    assert case["gold_chunks"] == []
    assert case["expect_status"] == "not_in_sources"
